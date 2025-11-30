////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"
#define CGLTF_IMPLEMENTATION 1
#include "cgltf.h"
#include <math.h>

// --- MATH HELPERS ---
typedef struct { f32 m[16]; } mat4;

// Matrix Multiplication: A * B (Column-Major)
mat4 mat4_mul(mat4 a, mat4 b) {
  mat4 r = {0};
  for(int col=0; col<4; ++col) {
    for(int row=0; row<4; ++row) {
      f32 sum = 0.0f;
      for(int k=0; k<4; ++k) {
        sum += a.m[k*4+row] * b.m[col*4+k];
      }
      r.m[col*4+row] = sum;
    }
  }
  return r;
}

// Transform (Translation, Rotation-Quaternion, Scale) to Mat4
mat4 transform_to_mat4(const f32* t, const f32* r, const f32* s) {
  f32 qx = r[0], qy = r[1], qz = r[2], qw = r[3];
  f32 tx = t[0], ty = t[1], tz = t[2];
  f32 sx = s[0], sy = s[1], sz = s[2];

  mat4 m;
  // Column 0
  m.m[0] = (1.0f - 2.0f*qy*qy - 2.0f*qz*qz) * sx;
  m.m[1] = (2.0f*qx*qy + 2.0f*qz*qw) * sx;
  m.m[2] = (2.0f*qx*qz - 2.0f*qy*qw) * sx;
  m.m[3] = 0.0f;

  // Column 1
  m.m[4] = (2.0f*qx*qy - 2.0f*qz*qw) * sy;
  m.m[5] = (1.0f - 2.0f*qx*qx - 2.0f*qz*qz) * sy;
  m.m[6] = (2.0f*qy*qz + 2.0f*qx*qw) * sy;
  m.m[7] = 0.0f;

  // Column 2
  m.m[8] = (2.0f*qx*qz + 2.0f*qy*qw) * sz;
  m.m[9] = (2.0f*qy*qz - 2.0f*qx*qw) * sz;
  m.m[10] = (1.0f - 2.0f*qx*qx - 2.0f*qy*qy) * sz;
  m.m[11] = 0.0f;

  // Column 3
  m.m[12] = tx; 
  m.m[13] = ty; 
  m.m[14] = tz; 
  m.m[15] = 1.0f;
  return m;
}

// --- ANIMATION SAMPLER ---

// Helper: Linear Interpolation for Vec3
void vec3_lerp(f32* out, const f32* a, const f32* b, f32 t) {
  out[0] = a[0] * (1.0f - t) + b[0] * t;
  out[1] = a[1] * (1.0f - t) + b[1] * t;
  out[2] = a[2] * (1.0f - t) + b[2] * t;
}

// Helper: Spherical Linear Interpolation for Quaternion
void quat_slerp(f32* out, const f32* a, const f32* b, f32 t) {
  f32 cosTheta = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
  f32 tempB[4];
  memcpy(tempB, b, 4 * sizeof(f32));

  // If dot is negative, flip one quaternion to take the shortest path
  if (cosTheta < 0.0f) {
    for(int i=0; i<4; i++) tempB[i] = -b[i];
    cosTheta = -cosTheta;
  }

  if (cosTheta > 0.9995f) {
    // If very close, just linear interpolate (avoid division by zero)
    for(int i=0; i<4; i++) out[i] = a[i] * (1.0f - t) + tempB[i] * t;
    // Normalize
    f32 len = sqrtf(out[0]*out[0] + out[1]*out[1] + out[2]*out[2] + out[3]*out[3]);
    for(int i=0; i<4; i++) out[i] /= len;
    return;
  }

  f32 angle = acosf(cosTheta);
  f32 sinAngle = sinf(angle);
  f32 w1 = sinf((1.0f - t) * angle) / sinAngle;
  f32 w2 = sinf(t * angle) / sinAngle;

  for(int i=0; i<4; i++) 
    out[i] = a[i] * w1 + tempB[i] * w2;
}

mat4 sample_node_at_time(cgltf_node* node, cgltf_animation* anim, f32 time) {
  f32 t[3]={0}, r[4]={0,0,0,1}, s[3]={1,1,1};
    
  // Default transforms
  if(node->has_translation) memcpy(t, node->translation, 3*sizeof(f32));
  if(node->has_rotation)    memcpy(r, node->rotation,    4*sizeof(f32));
  if(node->has_scale)       memcpy(s, node->scale,       3*sizeof(f32));

  if (anim) {
    for(int i=0; i<anim->channels_count; ++i) {
      cgltf_animation_channel* chan = &anim->channels[i];
      if(chan->target_node == node) {
        cgltf_animation_sampler* samp = chan->sampler;
        cgltf_accessor* input = samp->input;
        cgltf_accessor* output = samp->output;

        // 1. Find the Previous (k) and Next (k+1) frames
        int k = 0;
        for(; k < input->count - 1; ++k) {
          f32 t_curr, t_next;
          cgltf_accessor_read_float(input, k, &t_curr, 1);
          cgltf_accessor_read_float(input, k+1, &t_next, 1);
          if (time >= t_curr && time < t_next) break;
        }
                
        // 2. Calculate interpolation factor (0.0 to 1.0)
        f32 t0, t1;
        cgltf_accessor_read_float(input, k, &t0, 1);
        cgltf_accessor_read_float(input, (k+1 < input->count) ? k+1 : k, &t1, 1);
                
        f32 factor = 0.0f;
        if (t1 > t0) factor = (time - t0) / (t1 - t0);

        // 3. Read Values
        f32 val0[4], val1[4];
        int comp_count = (chan->target_path == cgltf_animation_path_type_rotation) ? 4 : 3;
                
        cgltf_accessor_read_float(output, k, val0, comp_count);
        cgltf_accessor_read_float(output, (k+1 < input->count) ? k+1 : k, val1, comp_count);

        // 4. Interpolate and Apply
        if(chan->target_path == cgltf_animation_path_type_translation) vec3_lerp(t, val0, val1, factor);
        if(chan->target_path == cgltf_animation_path_type_rotation)    quat_slerp(r, val0, val1, factor);
        if(chan->target_path == cgltf_animation_path_type_scale)       vec3_lerp(s, val0, val1, factor);
      }
    }
  }
  return transform_to_mat4(t, r, s);
}

// Recursive function to calculate Global World Transform
mat4 get_node_global_transform(cgltf_node* node, cgltf_animation* anim, f32 time) {
  mat4 local = sample_node_at_time(node, anim, time);
    
  if(node->parent) {
    mat4 parent_global = get_node_global_transform(node->parent, anim, time);
    return mat4_mul(parent_global, local);
  }
    
  return local;
}

// --- MAIN LOADER ---

// Returns array of 3 tensors: [0]=Vertices, [1]=Indices, [2]=Animation(or NULL)
// Output Animation Tensor Shape: [Frames, Bones, 4, 4]
tensor** loadGltfCooked(const char* filename, u32* outCount) {
  cgltf_options options = {0};
  cgltf_data* data = NULL;
    
  if (cgltf_parse_file(&options, filename, &data) != cgltf_result_success) 
    error("Failed to parse GLTF: %s", filename);
    
  if (cgltf_load_buffers(&options, data, filename) != cgltf_result_success) 
    error("Failed to load GLTF buffers: %s", filename);

  if (data->meshes_count == 0) error("No meshes found in %s", filename);

  // Setup Animation Baking
  tensor* t_anim = NULL;
  if (data->skins_count > 0 && data->animations_count > 0) {
    cgltf_skin* skin = &data->skins[0];
    cgltf_animation* anim = &data->animations[0];
    u32 bone_count = skin->joints_count;
    u32 fps = 30; // Bake rate
        
    f32 max_time = 0.0f;
    for(int i=0; i<anim->samplers_count; ++i) {
      if(anim->samplers[i].input->max[0] > max_time) 
        max_time = anim->samplers[i].input->max[0];
    }
        
    u32 frame_count = (u32)(max_time * fps) + 1;
    if(frame_count == 0) frame_count = 1;

    // Allocate Animation Tensor: [Frames, Bones, 4, 4]
    // Rank 4 tensor
    u32 anim_shape[4] = {frame_count, bone_count, 4, 4};
    f32* anim_data = mem(frame_count * bone_count * 16, f32);

    // BAKE LOOP
    for(u32 f=0; f<frame_count; ++f) {
      f32 time = (f32)f / (f32)fps;
            
      for(u32 b=0; b<bone_count; ++b) {
        cgltf_node* joint = skin->joints[b];
                
        // 1. Global Transform of Bone at Time T
        mat4 global_pose = get_node_global_transform(joint, anim, time);
                
        // 2. Inverse Bind Matrix (Bind Pose -> Bone Space)
        mat4 inv_bind = {0};
        // Default to identity if IBM not present
        inv_bind.m[0]=1; inv_bind.m[5]=1; inv_bind.m[10]=1; inv_bind.m[15]=1;
        if(skin->inverse_bind_matrices) {
          cgltf_accessor_read_float(skin->inverse_bind_matrices, b, inv_bind.m, 16);
        }

        // 3. Final Skin Matrix = Global * InvBind
        mat4 skin_mat = mat4_mul(global_pose, inv_bind);
                
        // 4. Store in Tensor (Row-major in memory for simple upload, or standard layout)
        // Tensor indexing: frame * (bones*16) + bone * 16
        u32 idx = (f * bone_count * 16) + (b * 16);
        memcpy(anim_data + idx, skin_mat.m, 16 * sizeof(f32));
      }
    }
        
    t_anim = newTensor(4, anim_shape, anim_data);
    tensorToGPUMemory(t_anim);
  } else {
    // Dummy animation tensor if none exists (1 frame, 1 bone, identity)
    u32 s[4] = {1,1,4,4};
    f32* d = mem(16, f32);
    memset(d, 0, 16*sizeof(f32));
    d[0]=1; d[5]=1; d[10]=1; d[15]=1;
    t_anim = newTensor(4, s, d);
    tensorToGPUMemory(t_anim);
  }

  // --- MESH LOADING (Vertices & Indices) ---
  // Layout: Pos(3) Norm(3) UV(2) Joints(4) Weights(4) MatID(1) = 17 floats
  u32 float_per_vert = 17;

  // First pass: count total vertices and indices across all meshes/primitives
  u32 total_verts = 0;
  u32 total_indices = 0;
  u32 total_prims = 0;

  for (u32 m = 0; m < data->meshes_count; ++m) {
    cgltf_mesh* mesh = &data->meshes[m];
    for (u32 p = 0; p < mesh->primitives_count; ++p) {
      cgltf_primitive* prim = &mesh->primitives[p];
      if (prim->attributes_count > 0) {
        total_verts += prim->attributes[0].data->count;
      }
      if (prim->indices) {
        total_indices += prim->indices->count;
      }
      total_prims++;
    }
  }

  if (total_verts == 0) error("No vertices found in %s", filename);

  // Allocate
  f32* vdata = mem(total_verts * float_per_vert, f32);
  memset(vdata, 0, total_verts * float_per_vert * sizeof(f32));

  f32* idata = NULL;
  if (total_indices > 0) {
    idata = mem(total_indices, f32);
  }

  // Second pass: fill data
  u32 vert_offset = 0;
  u32 index_offset = 0;
  u32 mat_id = 0;

  for (u32 m = 0; m < data->meshes_count; ++m) {
    cgltf_mesh* mesh = &data->meshes[m];
    for (u32 p = 0; p < mesh->primitives_count; ++p) {
      cgltf_primitive* prim = &mesh->primitives[p];
      if (prim->attributes_count == 0) continue;
        
      u32 prim_vert_count = prim->attributes[0].data->count;

      // Fill attributes for this primitive
      for (int i = 0; i < prim->attributes_count; ++i) {
        cgltf_attribute* attr = &prim->attributes[i];
        cgltf_accessor* acc = attr->data;
        int offset = -1;
        int num_comp = 0;

        if (attr->type == cgltf_attribute_type_position) { offset = 0;  num_comp = 3; }
        if (attr->type == cgltf_attribute_type_normal)   { offset = 3;  num_comp = 3; }
        if (attr->type == cgltf_attribute_type_texcoord) { offset = 6;  num_comp = 2; }
        if (attr->type == cgltf_attribute_type_joints)   { offset = 8;  num_comp = 4; }
        if (attr->type == cgltf_attribute_type_weights)  { offset = 12; num_comp = 4; }

        if (offset != -1) {
          for (u32 v = 0; v < prim_vert_count; ++v) {
            f32 temp[4] = {0};
            cgltf_accessor_read_float(acc, v, temp, num_comp);
            for (int c = 0; c < num_comp; ++c) {
              vdata[(vert_offset + v) * float_per_vert + offset + c] = temp[c];
            }
          }
        }
      }

      // Fill material ID for all verts in this primitive
      for (u32 v = 0; v < prim_vert_count; ++v) {
        vdata[(vert_offset + v) * float_per_vert + 16] = (f32)mat_id;
      }

      // Fill indices (offset by current vertex base)
      if (prim->indices) {
        for (u32 k = 0; k < prim->indices->count; ++k) {
          idata[index_offset + k] = (f32)(cgltf_accessor_read_index(prim->indices, k) + vert_offset);
        }
        index_offset += prim->indices->count;
      }

      vert_offset += prim_vert_count;
      mat_id++;
    }
  }

  // Create tensors
  u32 vshape[2] = {total_verts, float_per_vert};
  tensor* t_verts = newTensor(2, vshape, vdata);
  tensorToGPUMemory(t_verts);

  tensor* t_indices = NULL;
  if (total_indices > 0) {
    u32 ishape[1] = {total_indices};
    t_indices = newTensor(1, ishape, idata);
    tensorToGPUMemory(t_indices);
  }
  cgltf_free(data);

  // Pack Result
  tensor** result = mem(3, tensor*);
  result[0] = t_verts;   // Vertex Data (Pos, Norm, UV, Joints, Weights)
  result[1] = t_indices; // Index Data
  result[2] = t_anim;    // Animation Data
    
  if(outCount) *outCount = 3;
  return result;
}


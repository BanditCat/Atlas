////////////////////////////////////////////////////////////////////////////////
// Copyright © 2025 Jon DuBois. Written with the assistance of GPT-4 et al.   //
////////////////////////////////////////////////////////////////////////////////

#include "Atlas.h"
#define CGLTF_IMPLEMENTATION 1
#include "cgltf.h"
#include "stb_image.h"

#include <math.h>

// We use 2 layers per material: [0]=Albedo/Alpha, [1]=Normal/MR/AO
#define LAYERS_PER_MAT 2

// --- MATH HELPERS ---
typedef struct {
  f32 m[ 16 ];
} mat4;

// Matrix Multiplication: A * B (Column-Major)
mat4 mat4_mul( mat4 a, mat4 b ){
  mat4 r = { 0 };
  for( int col = 0; col < 4; ++col ){
    for( int row = 0; row < 4; ++row ){
      f32 sum = 0.0f;
      for( int k = 0; k < 4; ++k ){
        sum += a.m[ k * 4 + row ] * b.m[ col * 4 + k ];
      }
      r.m[ col * 4 + row ] = sum;
    }
  }
  return r;
}

// Transform (Translation, Rotation-Quaternion, Scale) to Mat4
mat4 transform_to_mat4( const f32* t, const f32* r, const f32* s ){
  f32 qx = r[ 0 ], qy = r[ 1 ], qz = r[ 2 ], qw = r[ 3 ];
  f32 tx = t[ 0 ], ty = t[ 1 ], tz = t[ 2 ];
  f32 sx = s[ 0 ], sy = s[ 1 ], sz = s[ 2 ];

  mat4 m;
  m.m[ 0 ] = ( 1.0f - 2.0f * qy * qy - 2.0f * qz * qz ) * sx;
  m.m[ 1 ] = ( 2.0f * qx * qy + 2.0f * qz * qw ) * sx;
  m.m[ 2 ] = ( 2.0f * qx * qz - 2.0f * qy * qw ) * sx;
  m.m[ 3 ] = 0.0f;

  m.m[ 4 ] = ( 2.0f * qx * qy - 2.0f * qz * qw ) * sy;
  m.m[ 5 ] = ( 1.0f - 2.0f * qx * qx - 2.0f * qz * qz ) * sy;
  m.m[ 6 ] = ( 2.0f * qy * qz + 2.0f * qx * qw ) * sy;
  m.m[ 7 ] = 0.0f;

  m.m[ 8 ] = ( 2.0f * qx * qz + 2.0f * qy * qw ) * sz;
  m.m[ 9 ] = ( 2.0f * qy * qz - 2.0f * qx * qw ) * sz;
  m.m[ 10 ] = ( 1.0f - 2.0f * qx * qx - 2.0f * qy * qy ) * sz;
  m.m[ 11 ] = 0.0f;

  m.m[ 12 ] = tx;
  m.m[ 13 ] = ty;
  m.m[ 14 ] = tz;
  m.m[ 15 ] = 1.0f;
  return m;
}

// --- ANIMATION SAMPLER ---
void vec3_lerp( f32* out, const f32* a, const f32* b, f32 t ){
  out[ 0 ] = a[ 0 ] * ( 1.0f - t ) + b[ 0 ] * t;
  out[ 1 ] = a[ 1 ] * ( 1.0f - t ) + b[ 1 ] * t;
  out[ 2 ] = a[ 2 ] * ( 1.0f - t ) + b[ 2 ] * t;
}

void quat_slerp( f32* out, const f32* a, const f32* b, f32 t ){
  f32 cosTheta =
    a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ] + a[ 3 ] * b[ 3 ];
  f32 tempB[ 4 ];
  memcpy( tempB, b, 4 * sizeof( f32 ) );

  if( cosTheta < 0.0f ){
    for( int i = 0; i < 4; i++ )
      tempB[ i ] = -b[ i ];
    cosTheta = -cosTheta;
  }

  if( cosTheta > 0.9995f ){
    for( int i = 0; i < 4; i++ )
      out[ i ] = a[ i ] * ( 1.0f - t ) + tempB[ i ] * t;
    f32 len = sqrtf( out[ 0 ] * out[ 0 ] + out[ 1 ] * out[ 1 ] +
                     out[ 2 ] * out[ 2 ] + out[ 3 ] * out[ 3 ] );
    for( int i = 0; i < 4; i++ )
      out[ i ] /= len;
    return;
  }

  f32 angle = acosf( cosTheta );
  f32 sinAngle = sinf( angle );
  f32 w1 = sinf( ( 1.0f - t ) * angle ) / sinAngle;
  f32 w2 = sinf( t * angle ) / sinAngle;

  for( int i = 0; i < 4; i++ )
    out[ i ] = a[ i ] * w1 + tempB[ i ] * w2;
}

mat4 sample_node_at_time( cgltf_node* node, cgltf_animation* anim, f32 time ){
  f32 t[ 3 ] = { 0 }, r[ 4 ] = { 0, 0, 0, 1 }, s[ 3 ] = { 1, 1, 1 };
  
  if( node->has_translation )
    memcpy( t, node->translation, 3 * sizeof( f32 ) );
  if( node->has_rotation )
    memcpy( r, node->rotation, 4 * sizeof( f32 ) );
  /* if( node->has_scale ){ */
  /*   memcpy( s, node->scale, 3 * sizeof( f32 ) ); */
  /*   printf("boop! %f %f %f\n", node->scale[ 0 ], node->scale[ 1 ], node->scale[ 2 ] ); */
  /* } */

  if( anim ){
    for( int i = 0; i < anim->channels_count; ++i ){
      cgltf_animation_channel* chan = &anim->channels[ i ];
      if( chan->target_node == node ){
        cgltf_animation_sampler* samp = chan->sampler;
        cgltf_accessor* input = samp->input;
        cgltf_accessor* output = samp->output;

        int k = 0;
        for( ; k < input->count - 1; ++k ){
          f32 t_curr, t_next;
          cgltf_accessor_read_float( input, k, &t_curr, 1 );
          cgltf_accessor_read_float( input, k + 1, &t_next, 1 );
          if( time >= t_curr && time < t_next )
            break;
        }

        f32 t0, t1;
        cgltf_accessor_read_float( input, k, &t0, 1 );
        cgltf_accessor_read_float(
                                  input, ( k + 1 < input->count ) ? k + 1 : k, &t1, 1 );

        f32 factor = 0.0f;
        if( t1 > t0 )
          factor = ( time - t0 ) / ( t1 - t0 );

        f32 val0[ 4 ], val1[ 4 ];
        int comp_count =
          ( chan->target_path == cgltf_animation_path_type_rotation ) ? 4 : 3;

        cgltf_accessor_read_float( output, k, val0, comp_count );
        cgltf_accessor_read_float(
                                  output, ( k + 1 < input->count ) ? k + 1 : k, val1, comp_count );
        if( chan->target_path == cgltf_animation_path_type_translation )
          vec3_lerp( t, val0, val1, factor );
        if( chan->target_path == cgltf_animation_path_type_rotation )
          quat_slerp( r, val0, val1, factor );
        if( chan->target_path == cgltf_animation_path_type_scale )
          vec3_lerp( s, val0, val1, factor );
      }
    }
  }
  return transform_to_mat4( t, r, s );
}

mat4 get_node_global_transform( cgltf_node* node,
                                cgltf_animation* anim,
                                f32 time ){

  mat4 local = sample_node_at_time( node, anim, time );
  if( node->parent ){
    // Pass scale to parent
    mat4 parent_global = get_node_global_transform( node->parent, anim, time );
    return mat4_mul( parent_global, local );
  }
  return local;
}

// --- TEXTURE HELPER ---
// Resizes image using linear interpolation.
// Input: Raw bytes (channels count varies)
// Output: Always RGBA (4 bytes per pixel)
void resize_image(
                  u8* src, u32 sw, u32 sh, u32 dw, u32 dh, u32 channels, f32* dst ){
  float x_ratio = (float)sw / (float)dw;
  float y_ratio = (float)sh / (float)dh;

  // Downsampling: box filter (average all source pixels in footprint)
  // Upsampling: bilinear interpolation
  int downsample = ( x_ratio > 1.0f ) || ( y_ratio > 1.0f );

  for( u32 y = 0; y < dh; y++ ){
    for( u32 x = 0; x < dw; x++ ){
      float accum[ 4 ] = { 0, 0, 0, 0 };
      u32 dst_idx = ( y * dw + x ) * 4;

      if( downsample ){
        // Box filter: average all source pixels covered by this dest pixel
        float sx0 = x * x_ratio;
        float sy0 = y * y_ratio;
        float sx1 = ( x + 1 ) * x_ratio;
        float sy1 = ( y + 1 ) * y_ratio;

        u32 ix0 = (u32)sx0;
        u32 iy0 = (u32)sy0;
        u32 ix1 = (u32)ceilf( sx1 );
        u32 iy1 = (u32)ceilf( sy1 );
        if( ix1 > sw )
          ix1 = sw;
        if( iy1 > sh )
          iy1 = sh;

        float total_weight = 0.0f;

        for( u32 sy = iy0; sy < iy1; sy++ ){
          float wy = 1.0f;
          if( sy < sy0 )
            wy -= ( sy0 - sy );
          if( sy + 1 > sy1 )
            wy -= ( ( sy + 1 ) - sy1 );

          for( u32 sx = ix0; sx < ix1; sx++ ){
            float wx = 1.0f;
            if( sx < sx0 )
              wx -= ( sx0 - sx );
            if( sx + 1 > sx1 )
              wx -= ( ( sx + 1 ) - sx1 );

            float w = wx * wy;
            total_weight += w;

            u32 src_idx = ( sy * sw + sx ) * channels;
            for( u32 c = 0; c < 4; c++ ){
              u8 val;
              if( c < channels )
                val = src[ src_idx + c ];
              else if( c == 3 )
                val = 255;
              else
                val = src[ src_idx ];
              accum[ c ] += w * val;
            }
          }
        }

        for( u32 c = 0; c < 4; c++ )
          dst[ dst_idx + c ] = ( accum[ c ] / total_weight ) / 255.0f;

      } else {
        // Bilinear interpolation for upsampling
        float sx = ( x + 0.5f ) * x_ratio - 0.5f;
        float sy = ( y + 0.5f ) * y_ratio - 0.5f;

        if( sx < 0 )
          sx = 0;
        if( sy < 0 )
          sy = 0;

        u32 x0 = (u32)sx;
        u32 y0 = (u32)sy;
        u32 x1 = ( x0 + 1 < sw ) ? x0 + 1 : x0;
        u32 y1 = ( y0 + 1 < sh ) ? y0 + 1 : y0;

        float fx = sx - x0;
        float fy = sy - y0;

        float w00 = ( 1.0f - fx ) * ( 1.0f - fy );
        float w10 = fx * ( 1.0f - fy );
        float w01 = ( 1.0f - fx ) * fy;
        float w11 = fx * fy;

        for( u32 c = 0; c < 4; c++ ){
          u8 v00, v10, v01, v11;

          if( c < channels ){
            v00 = src[ ( y0 * sw + x0 ) * channels + c ];
            v10 = src[ ( y0 * sw + x1 ) * channels + c ];
            v01 = src[ ( y1 * sw + x0 ) * channels + c ];
            v11 = src[ ( y1 * sw + x1 ) * channels + c ];
          } else if( c == 3 ){
            v00 = v10 = v01 = v11 = 255;
          } else {
            v00 = src[ ( y0 * sw + x0 ) * channels ];
            v10 = src[ ( y0 * sw + x1 ) * channels ];
            v01 = src[ ( y1 * sw + x0 ) * channels ];
            v11 = src[ ( y1 * sw + x1 ) * channels ];
          }

          accum[ c ] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
        }

        for( u32 c = 0; c < 4; c++ )
          dst[ dst_idx + c ] = accum[ c ] / 255.0f;
      }
    }
  }
}

// --- MAIN LOADER ---
// Returns [0]=Vertices, [1]=Indices, [2]=Animation, [3]=TextureArray
tensor** loadGltfCooked( const char* filename, u32* outCount ){
  f32 animCount;
  
  // 1. FILE I/O (Manual to avoid fopen issues in some emscripten setups)
  FILE* file = fopen( filename, "rb" );
  if( !file )
    error( "ATLAS: Could not open file: %s", filename );
  fseek( file, 0, SEEK_END );
  long fileSize = ftell( file );
  fseek( file, 0, SEEK_SET );
  if( fileSize <= 0 )
    error( "ATLAS: Empty file: %s", filename );
  void* fileData = mem( fileSize, u8 );
  fread( fileData, 1, fileSize, file );
  fclose( file );


  
  // 2. PARSE GLTF
  cgltf_options options = { 0 };
  cgltf_data* data = NULL;
  // Parse from memory buffer


  if( cgltf_parse( &options, fileData, fileSize, &data ) !=
      cgltf_result_success )
    error( "%s", "cgltf_parse failed" );
  // Load buffers (embedded images/bin)
  if( cgltf_load_buffers( &options, data, filename ) != cgltf_result_success )
    error( "%s", "load buffers failed" );
  if( data->meshes_count == 0 )
    error( "%s", "No meshes" );
  // IMMEDIATELY after parse - before any material/texture loops
  for (u32 m = 0; m < data->meshes_count; ++m) {
    cgltf_mesh* mesh = &data->meshes[m];
    printf("Mesh %u: prims=%zu\n", m, mesh->primitives_count);
    for (u32 p = 0; p < mesh->primitives_count; ++p) {
      cgltf_primitive* prim = &mesh->primitives[p];
      printf("  prim %u: type=%d\n", p, prim->type);
    }
  }
  for (u32 i = 0; i < data->materials_count; ++i) {
    cgltf_material* mat = &data->materials[i];
    printf("Mat %d: %s\n", i, mat->name ? mat->name : "unnamed");
    
    f32* f = mat->pbr_metallic_roughness.base_color_factor;
    printf("  base_color: %.2f %.2f %.2f %.2f\n", f[0], f[1], f[2], f[3]);
    printf("  emissive: %.2f %.2f %.2f\n", 
           mat->emissive_factor[0], 
           mat->emissive_factor[1], 
           mat->emissive_factor[2]);
    if (mat->emissive_texture.texture) printf("  has emissive texture\n");
    if (mat->has_pbr_specular_glossiness) printf("  has spec/gloss\n");
    if (mat->unlit) printf("  UNLIT material\n");
  }

  // In the mesh loop, check for vertex colors:
  for (u32 m = 0; m < data->meshes_count; ++m) {
    cgltf_mesh* mesh = &data->meshes[m];
    printf("Mesh %d: %s\n", m, mesh->name ? mesh->name : "unnamed");
    for (u32 p = 0; p < mesh->primitives_count; ++p) {
      cgltf_primitive* prim = &mesh->primitives[p];
      for (int i = 0; i < prim->attributes_count; ++i) {
        cgltf_attribute* attr = &prim->attributes[i];
        if (attr->type == cgltf_attribute_type_color) {
          printf("  prim %d HAS VERTEX COLORS\n", p);
        }
      }
    }
  }
  for (u32 i = 0; i < data->materials_count; ++i) {
    cgltf_material* mat = &data->materials[i];
    cgltf_texture* albedo_tex = mat->pbr_metallic_roughness.base_color_texture.texture;
    cgltf_texture* pbr_tex = mat->pbr_metallic_roughness.metallic_roughness_texture.texture;
    
    printf("Mat %d (%s):\n", i, mat->name ? mat->name : "unnamed");
    printf("  albedo_tex=%p  image=%p\n", 
           (void*)albedo_tex, 
           albedo_tex ? (void*)albedo_tex->image : NULL);
    printf("  pbr_tex=%p  image=%p\n", 
           (void*)pbr_tex, 
           pbr_tex ? (void*)pbr_tex->image : NULL);
    
    if (albedo_tex && pbr_tex && albedo_tex->image == pbr_tex->image) {
      printf("  WARNING: SAME IMAGE FOR ALBEDO AND PBR!\n");
    }
  }


  // --- FIND MAX TEXTURE DIMENSIONS ---
  u32 TEX_W = 512;  // default fallback
  u32 TEX_H = 512;

  for (u32 i = 0; i < data->images_count; ++i) {
    cgltf_image* img = &data->images[i];
    if (img->buffer_view) {
      int w, h, c;
      if (stbi_info_from_memory(
                                (u8*)img->buffer_view->buffer->data + img->buffer_view->offset,
                                img->buffer_view->size, &w, &h, &c)) {
        if ((u32)w > TEX_W) TEX_W = (u32)w;
        if ((u32)h > TEX_H) TEX_H = (u32)h;
      }
    }
  }
  printf("Texture atlas size: %d x %d\n", TEX_W, TEX_H);

 
  // --- ANIMATION BAKING ---
  tensor* t_anim = NULL;
  // (Standard animation code from before)

  
  if( data->skins_count > 0 && data->animations_count > 0 ){
    cgltf_skin* skin = &data->skins[ 0 ];
    u32 bone_count = skin->joints_count;
    u32 anim_count = data->animations_count;
    animCount = anim_count;
    u32 fps = 24;

    f32 global_max_time = 0.0f;
    for( u32 a = 0; a < anim_count; ++a ){
      cgltf_animation* anim = &data->animations[ a ];
      for( int i = 0; i < anim->samplers_count; ++i ){
        if( anim->samplers[ i ].input->max[ 0 ] > global_max_time )
          global_max_time = anim->samplers[ i ].input->max[ 0 ];
      }
    }

    u32 frame_count = (u32)( global_max_time * fps ) + 1;
    if( frame_count == 0 )
      frame_count = 1;

    u32 anim_shape[ 4 ] = { frame_count, bone_count * anim_count, 4, 4 };
    f32* anim_data = mem( frame_count * bone_count * anim_count * 16, f32 );

    for( u32 f = 0; f < frame_count; ++f ){
      f32 time = (f32)f / (f32)fps;
      for( u32 b = 0; b < bone_count; ++b ){
        cgltf_node* joint = skin->joints[ b ];
        mat4 inv_bind = { 0 };
        inv_bind.m[ 0 ] = 1;
        inv_bind.m[ 5 ] = 1;
        inv_bind.m[ 10 ] = 1;
        inv_bind.m[ 15 ] = 1;
        if( skin->inverse_bind_matrices )
          cgltf_accessor_read_float(
                                    skin->inverse_bind_matrices, b, inv_bind.m, 16 );

        for( u32 a = 0; a < anim_count; ++a ){
          mat4 global_pose =
            get_node_global_transform( joint, &data->animations[ a ], time );
          mat4 skin_mat = mat4_mul( global_pose, inv_bind );
          u32 idx = ( f * bone_count * anim_count * 16 ) +
            ( b * anim_count * 16 ) + ( a * 16 );
          memcpy( anim_data + idx, skin_mat.m, 16 * sizeof( f32 ) );
        }
      }
    }
    t_anim = newTensor( 4, anim_shape, anim_data );
    tensorToTextureArrayOld( t_anim, 4 );
  } else {
    // Dummy animation
    u32 s[ 4 ] = { 1, 1, 1, 16 };
    f32* d = mem( 16, f32 );
    memset( d, 0, 16 * sizeof( f32 ) );
    d[ 0 ] = 1;
    d[ 5 ] = 1;
    d[ 10 ] = 1;
    d[ 15 ] = 1;
    t_anim = newTensor( 4, s, d );
    tensorToGPUMemory( t_anim );
  }

  // --- TEXTURE BAKING (THE ATLAS PACKER) ---
  tensor* t_tex = NULL;
  u32 num_mats = data->materials_count;
  if( num_mats == 0 )
    num_mats = 1;

  // Tensor Shape: [Total Layers, Height, Width, RGBA]
  // Note: Most texture arrays are [Layer, H, W, 4], make sure your
  // tensorToTextureArrayOld expects this.
  u32 tex_shape[ 4 ] = { num_mats * LAYERS_PER_MAT, TEX_H, TEX_W, 4 };
  u32 total_pixels = ( num_mats * LAYERS_PER_MAT ) * TEX_W * TEX_H;
  f32* tex_data = mem( total_pixels * 4, f32 );

  // 1. Initialize with Defaults
  for( u32 i = 0; i < num_mats; ++i ){
    f32* l0 = tex_data + ( ( i * 2 + 0 ) * TEX_W * TEX_H * 4 );  // Layer 0: Vis
    f32* l1 = tex_data + ( ( i * 2 + 1 ) * TEX_W * TEX_H * 4 );  // Layer 1: Phy

    for( int p = 0; p < TEX_W * TEX_H; ++p ){
      // L0 Default: White Albedo, Full Opacity
      l0[ p * 4 + 0 ] = 1.0;
      l0[ p * 4 + 1 ] = 1.0;
      l0[ p * 4 + 2 ] = 1.0;
      l0[ p * 4 + 3 ] = 1.0;
      // L1 Default: Flat Normal(128,128,255), Rough(255), Occ(255)
      l1[ p * 4 + 0 ] = 0.5;
      l1[ p * 4 + 1 ] = 0.5;
      l1[ p * 4 + 2 ] = 1.0;
      l1[ p * 4 + 3 ] = 1.0;
    }
  }

  // 2. Process Materials
  for( u32 i = 0; i < data->materials_count; ++i ){
    cgltf_material* mat = &data->materials[ i ];
    f32* l0 = tex_data + ( ( i * 2 + 0 ) * TEX_W * TEX_H * 4 );
    f32* l1 = tex_data + ( ( i * 2 + 1 ) * TEX_W * TEX_H * 4 );

    if (mat->pbr_metallic_roughness.base_color_texture.texture) {
      printf("Mat %d uses texcoord %d\n", i, 
             mat->pbr_metallic_roughness.base_color_texture.texcoord);
    }
    
    // --- LAYER 0: ALBEDO (RGB) + ALPHA (A) ---
    if( mat->pbr_metallic_roughness.base_color_texture.texture ){
      cgltf_image* img =
        mat->pbr_metallic_roughness.base_color_texture.texture->image;
      // Only handle BufferViews (embedded) for now. External URI needs separate
      // load.
      if( img && img->buffer_view ){
        int w, h, c;
        u8* raw = stbi_load_from_memory( (u8*)img->buffer_view->buffer->data +
                                         img->buffer_view->offset,
                                         img->buffer_view->size,
                                         &w,
                                         &h,
                                         &c,
                                         0 );


        if (raw) {
          printf("  Loaded albedo for mat %d: %dx%d, %d channels\n", i, w, h, c);
          resize_image(raw, w, h, TEX_W, TEX_H, c, l0);
          free(raw);
        } else {
          printf("  FAILED to load albedo for mat %d\n", i);
        }

        
        
      }
    }
    f32* factor = mat->pbr_metallic_roughness.base_color_factor;
    for (int p = 0; p < TEX_W * TEX_H; ++p) {
      // Convert linear factor to sRGB before multiplying
      l0[p*4+0] *= powf(factor[0], 1.0f/2.2f);
      l0[p*4+1] *= powf(factor[1], 1.0f/2.2f);
      l0[p*4+2] *= powf(factor[2], 1.0f/2.2f);
      l0[p*4+3] *= 1.0;//factor[3];  // alpha stays linear
    }

    if (mat->alpha_mode == cgltf_alpha_mode_opaque) {
      // Force alpha to 1.0
      for (int p = 0; p < TEX_W * TEX_H; ++p) {
        l0[p*4+3] = 1.0f;
      }
    }
    // --- LAYER 1 PREP: LOAD SOURCE IMAGES ---

    // Load NORMAL (into L1 RGBA temporarily)
    if( mat->normal_texture.texture ){
      cgltf_image* img = mat->normal_texture.texture->image;
      if( img && img->buffer_view ){
        int w, h, c;
        u8* raw = stbi_load_from_memory( (u8*)img->buffer_view->buffer->data +
                                         img->buffer_view->offset,
                                         img->buffer_view->size,
                                         &w,
                                         &h,
                                         &c,
                                         0 );
        if( raw ){
          resize_image(
                       raw, w, h, TEX_W, TEX_H, c, l1 );  // L1 now holds Normal RGB
          free( raw );
        }
      }
    }

    // Load ORM / PBR (Metallic=B, Roughness=G)
    f32* pbr_pixels = NULL;
    if( mat->pbr_metallic_roughness.metallic_roughness_texture.texture ){
      cgltf_image* img =
        mat->pbr_metallic_roughness.metallic_roughness_texture.texture->image;
      if( img && img->buffer_view ){
        int w, h, c;
        u8* raw = stbi_load_from_memory( (u8*)img->buffer_view->buffer->data +
                                         img->buffer_view->offset,
                                         img->buffer_view->size,
                                         &w,
                                         &h,
                                         &c,
                                         0 );
        if( raw ){
          pbr_pixels = mem( TEX_W * TEX_H * 4, f32 );
          resize_image( raw, w, h, TEX_W, TEX_H, c, pbr_pixels );
          free( raw );
        }
      }
    }

    // Load OCCLUSION (R channel)
    f32* occ_pixels = NULL;
    if( mat->occlusion_texture.texture ){
      cgltf_image* img = mat->occlusion_texture.texture->image;
      if( img && img->buffer_view ){
        int w, h, c;
        u8* raw = stbi_load_from_memory( (u8*)img->buffer_view->buffer->data +
                                         img->buffer_view->offset,
                                         img->buffer_view->size,
                                         &w,
                                         &h,
                                         &c,
                                         0 );
        if( raw ){
          occ_pixels = mem( TEX_W * TEX_H * 4, f32 );
          resize_image( raw, w, h, TEX_W, TEX_H, c, occ_pixels );
          free( raw );
        }
      }
    }

    for( int p = 0; p < TEX_W * TEX_H; ++p ){
      f32* px = l1 + ( p * 4 );

      // px[0] and px[1] already have Normal X, Y as floats (0-1)

      // --- PACK B: Metal (2 bits) | Rough (6 bits) ---
      u8 m_val = 0;  // Default Dielectric
      u8 r_val = 0;  // Default Rough

      if( pbr_pixels ){
        // Standard glTF: Metallic is B, Roughness is G (now floats 0-1)
        m_val = (u8)( pbr_pixels[ p * 4 + 2 ] * 255.0f );
        r_val = (u8)( pbr_pixels[ p * 4 + 1 ] * 255.0f );
      }

      u8 m_bits = m_val & 0xC0;                     // Top 2 bits
      u8 r_bits = ( r_val >> 2 ) & 0x3F;            // Top 6 bits shifted down
      px[ 2 ] = (f32)( m_bits | r_bits ) / 255.0f;  // Store in Blue, normalized

      // --- PACK A: Occlusion ---
      if( occ_pixels ){
        px[ 3 ] = occ_pixels[ p * 4 + 0 ];  // Already float 0-1
      } else {
        px[ 3 ] = 1.0f;  // No occlusion (White)
      }
    }
    if( pbr_pixels )
      unmem( pbr_pixels );
    if( occ_pixels )
      unmem( occ_pixels );
  }


  printf( "foof\n" );
  t_tex = newTensor( 4, tex_shape, (f32*)tex_data );
  tensorToTextureArrayOld( t_tex, 40 );
  t_tex->tex.mipmapped = 1;
  printf( "foo\n" );
  textureTensor( t_tex );
  printf( "bar\n" );
  
  // --- MESH LOADING (Vertices & Indices) ---
  // (Standard Mesh code...)
  u32 float_per_vert = 21;
  u32 total_verts = 0;
  u32 total_indices = 0;

  for( u32 m = 0; m < data->meshes_count; ++m ){
    cgltf_mesh* mesh = &data->meshes[ m ];
    for( u32 p = 0; p < mesh->primitives_count; ++p ){
      if( mesh->primitives[ p ].attributes_count > 0 )
        total_verts += mesh->primitives[ p ].attributes[ 0 ].data->count;
      if( mesh->primitives[ p ].indices )
        total_indices += mesh->primitives[ p ].indices->count;
    }
  }

  if( total_verts == 0 )
    error( "%s", "No vertices" );

  f32* vdata = mem( total_verts * float_per_vert, f32 );

  // --- CHEDDAR: AUTOMATIC UNIT SCALING ---
  f32 global_scale = 1.0f;
  f32 max_coord = 0.0f;

  // Scan all accessors to find the bounds of the model
  for (u32 m = 0; m < data->meshes_count; ++m) {
    cgltf_mesh* mesh = &data->meshes[m];
    for (u32 p = 0; p < mesh->primitives_count; ++p) {
      cgltf_primitive* prim = &mesh->primitives[p];
      for (int i = 0; i < prim->attributes_count; ++i) {
        cgltf_attribute* attr = &prim->attributes[i];
        if (attr->type == cgltf_attribute_type_position && attr->data->has_max) {
          // Check X, Y, Z max bounds
          for(int c=0; c<3; ++c) {
            f32 val = fabsf(attr->data->max[c]);
            if (val > max_coord) max_coord = val;
            // Check min too (for negative extents)
            val = fabsf(attr->data->min[c]);
            if (val > max_coord) max_coord = val;
          }
        }
      }
    }
  }

  // If the model is massive (e.g. > 100 units), assume CM and scale down
  if (max_coord > 50.0f) {
    printf("ATLAS: Model is huge (max coord: %.2f). Assuming CM -> Scaling by 0.01\n", max_coord);
    global_scale = 0.01f;
  }
  
  f32* idata = ( total_indices > 0 ) ? mem( total_indices, f32 ) : NULL;

  u32 vert_offset = 0;
  u32 index_offset = 0;

  for( u32 m = 0; m < data->meshes_count; ++m ){
    cgltf_mesh* mesh = &data->meshes[ m ];
    for( u32 p = 0; p < mesh->primitives_count; ++p ){
      cgltf_primitive* prim = &mesh->primitives[ p ];
      if( prim->attributes_count == 0 )
        continue;
      u32 prim_vert_count = prim->attributes[ 0 ].data->count;


      for (int i = 0; i < prim->attributes_count; ++i) {
        cgltf_attribute* attr = &prim->attributes[i];
        if (attr->type == cgltf_attribute_type_texcoord) {
          printf("Mesh %s prim %d has TEXCOORD_%d\n", 
                 mesh->name ? mesh->name : "unnamed", p, attr->index);
        }
      }

 
      for( int i = 0; i < prim->attributes_count; ++i ){
        cgltf_attribute* attr = &prim->attributes[ i ];
        int offset = -1;
        int num_comp = 0;
        if( attr->type == cgltf_attribute_type_position ){
          offset = 0;
          num_comp = 3;
        }
        if( attr->type == cgltf_attribute_type_normal ){
          offset = 3;
          num_comp = 3;
        }
        if( attr->type == cgltf_attribute_type_tangent ){
          offset = 17;
          num_comp = 4;
        }
        if( attr->type == cgltf_attribute_type_texcoord ){

          for( u32 v = 0; v < prim_vert_count; ++v ){
            f32 temp[ 2 ] = { 0 };
            cgltf_accessor_read_float( attr->data, v, temp, 2 );
            vdata[ ( vert_offset + v ) * float_per_vert + 6 ] = fmaxf(0.0f, fminf(1.0f, temp[0]));
            vdata[ ( vert_offset + v ) * float_per_vert + 7 ] = fmaxf(0.0f, fminf(1.0f, temp[1]));
          }


          if (attr->type == cgltf_attribute_type_texcoord && attr->index == 0) {
            for (u32 v = 0; v < attr->data->count; ++v) {
              f32 temp[2];
              cgltf_accessor_read_float(attr->data, v, temp, 2);
              u32 base = (vert_offset + v) * float_per_vert;
              vdata[base + 6] = temp[0];//fmaxf(0.0f, fminf(1.0f, temp[0]));
              vdata[base + 7] = temp[1];//fmaxf(0.0f, fminf(1.0f, temp[1]));
        
              // Debug: confirm writes are happening
              if (v < 2 && strstr(mesh->name, "eye")) {
                printf("  WROTE UV[%d] at base=%d: (%.4f, %.4f)\n", 
                       v, base, vdata[base + 6], vdata[base + 7]);
              }
            }
          }

        }
        if( attr->type == cgltf_attribute_type_joints ){
          offset = 8;
          num_comp = 4;
        }
        if( attr->type == cgltf_attribute_type_weights ){
          offset = 12;
          num_comp = 4;
        }


        if (attr->type == cgltf_attribute_type_texcoord && attr->index == 0) {
          printf("  TEXCOORD_0 accessor: count=%d, component_type=%d, type=%d\n",
                 (int)attr->data->count, attr->data->component_type, attr->data->type);
    
          // Read raw first few values
          for (u32 v = 0; v < 3 && v < attr->data->count; ++v) {
            f32 temp[2] = {-999, -999};
            cgltf_accessor_read_float(attr->data, v, temp, 2);
            printf("    raw uv[%d] = (%.4f, %.4f)\n", v, temp[0], temp[1]);
          }
        }
        
        if( offset != -1 ){
          for( u32 v = 0; v < prim_vert_count; ++v ){
            f32 temp[ 4 ] = { 0 };
            cgltf_accessor_read_float( attr->data, v, temp, num_comp );
            
            if (attr->type == cgltf_attribute_type_position) {
              temp[0] *= global_scale;
              temp[1] *= global_scale;
              temp[2] *= global_scale;
            }

            for( int c = 0; c < num_comp; ++c )
              vdata[ ( vert_offset + v ) * float_per_vert + offset + c ] =
                temp[ c ];
          }
        }
      }



      // ==================================================================================
      // === TANGENT GENERATOR (The "Anti-Void" Logic) ====================================
      // ==================================================================================
      
      // 1. Check if we actually found tangents in the file
      bool found_tangents = false;
      for (int i = 0; i < prim->attributes_count; ++i) {
        if (prim->attributes[i].type == cgltf_attribute_type_tangent) {
            found_tangents = true;
            break;
        }
      }

      // 2. If NO tangents, we must calculate them!
      if (!found_tangents && prim_vert_count > 0) {
          printf("ATLAS: Generating missing tangents for mesh %d prim %d...\n", m, p);

          // Local struct for cleaner math
          typedef struct { f32 x, y, z; } vec3;

          vec3* temp_tan = mem(prim_vert_count, vec3); 
          vec3* temp_bit = mem(prim_vert_count, vec3);
          memset(temp_tan, 0, prim_vert_count * sizeof(vec3));
          memset(temp_bit, 0, prim_vert_count * sizeof(vec3));

          // A. Iterate over Indices (Triangles)
          u32 index_count = (prim->indices) ? prim->indices->count : prim_vert_count;
          
          for (u32 k = 0; k < index_count; k += 3) {
              u32 i0, i1, i2;
              if (prim->indices) {
                  i0 = cgltf_accessor_read_index(prim->indices, k + 0);
                  i1 = cgltf_accessor_read_index(prim->indices, k + 1);
                  i2 = cgltf_accessor_read_index(prim->indices, k + 2);
              } else {
                  i0 = k + 0; i1 = k + 1; i2 = k + 2;
              }

              // Read Pos (Offset 0) and UV (Offset 6)
              // We use float_per_vert (which should be 21 now)
              u32 stride = float_per_vert;
              f32* v0 = &vdata[(vert_offset + i0) * stride];
              f32* v1 = &vdata[(vert_offset + i1) * stride];
              f32* v2 = &vdata[(vert_offset + i2) * stride];

              // Pos at offset 0
              f32 x1 = v1[0] - v0[0]; f32 x2 = v2[0] - v0[0];
              f32 y1 = v1[1] - v0[1]; f32 y2 = v2[1] - v0[1];
              f32 z1 = v1[2] - v0[2]; f32 z2 = v2[2] - v0[2];

              // UV at offset 6
              f32 s1 = v1[6] - v0[6]; f32 s2 = v2[6] - v0[6];
              f32 t1 = v1[7] - v0[7]; f32 t2 = v2[7] - v0[7];

              f32 r = 1.0f / (s1 * t2 - s2 * t1);
              
              // Prevent div by zero 
              if (isinf(r) || isnan(r)) r = 0.0f;

              f32 sdir[3] = { (t2 * x1 - t1 * x2) * r, 
                              (t2 * y1 - t1 * y2) * r, 
                              (t2 * z1 - t1 * z2) * r };
                              
              f32 tdir[3] = { (s1 * x2 - s2 * x1) * r, 
                              (s1 * y2 - s2 * y1) * r, 
                              (s1 * z2 - s2 * z1) * r };

              // Accumulate
              temp_tan[i0].x += sdir[0]; temp_tan[i0].y += sdir[1]; temp_tan[i0].z += sdir[2];
              temp_tan[i1].x += sdir[0]; temp_tan[i1].y += sdir[1]; temp_tan[i1].z += sdir[2];
              temp_tan[i2].x += sdir[0]; temp_tan[i2].y += sdir[1]; temp_tan[i2].z += sdir[2];

              temp_bit[i0].x += tdir[0]; temp_bit[i0].y += tdir[1]; temp_bit[i0].z += tdir[2];
              temp_bit[i1].x += tdir[0]; temp_bit[i1].y += tdir[1]; temp_bit[i1].z += tdir[2];
              temp_bit[i2].x += tdir[0]; temp_bit[i2].y += tdir[1]; temp_bit[i2].z += tdir[2];
          }

          // B. Orthogonalize and Write to vdata
          for (u32 i = 0; i < prim_vert_count; ++i) {
               u32 stride = float_per_vert;
               f32* vert = &vdata[(vert_offset + i) * stride];
               
               // Read Normal (Offset 3)
               f32 n[3] = { vert[3], vert[4], vert[5] };
               f32 t[3] = { temp_tan[i].x, temp_tan[i].y, temp_tan[i].z };
               
               // Gram-Schmidt Orthogonalize: T = T - N * dot(N, T)
               f32 dotNT = n[0]*t[0] + n[1]*t[1] + n[2]*t[2];
               f32 ortho[3] = { t[0] - n[0]*dotNT, 
                                t[1] - n[1]*dotNT, 
                                t[2] - n[2]*dotNT };
               
               // Normalize
               f32 len = sqrtf(ortho[0]*ortho[0] + ortho[1]*ortho[1] + ortho[2]*ortho[2]);
               if (len > 0.0f) {
                   ortho[0] /= len; ortho[1] /= len; ortho[2] /= len;
               }

               // Calculate Handedness (W component)
               // Cross(N, T) dot Bitangent
               f32 b[3] = { temp_bit[i].x, temp_bit[i].y, temp_bit[i].z };
               f32 c[3] = { n[1]*t[2] - n[2]*t[1], 
                            n[2]*t[0] - n[0]*t[2], 
                            n[0]*t[1] - n[1]*t[0] };
               
               f32 val = c[0]*b[0] + c[1]*b[1] + c[2]*b[2];
               f32 w = (val < 0.0f) ? -1.0f : 1.0f;

               // Store at Offset 17 (Tangent XYZW)
               vert[17] = ortho[0];
               vert[18] = ortho[1];
               vert[19] = ortho[2];
               vert[20] = w;
          }

          unmem(temp_tan);
          unmem(temp_bit);
      }
      // ==================================================================================



      u32 prim_mat_id = 0;
 
      if (prim->material) {
        for (u32 mi = 0; mi < data->materials_count; mi++) {
          if (prim->material == &data->materials[mi]) {
            prim_mat_id = mi;
            break;
          }
        }
      }

      for (u32 v = 0; v < prim_vert_count; ++v)
        vdata[(vert_offset + v) * float_per_vert + 16] = (f32)prim_mat_id;
      if( prim->indices ){
        for( u32 k = 0; k < prim->indices->count; ++k ){
          idata[ index_offset + k ] =
            (f32)( cgltf_accessor_read_index( prim->indices, k ) +
                   vert_offset );
        }
        index_offset += prim->indices->count;
      }

      // Find parent bone for unskinned meshes
      cgltf_node* mesh_node = NULL;
      for (u32 n = 0; n < data->nodes_count; ++n) {
        if (data->nodes[n].mesh == mesh) {
          mesh_node = &data->nodes[n];
          break;
        }
      }

      // Find which bone this node is parented to
      int parent_bone_idx = -1;
      if (mesh_node && data->skins_count > 0) {
        cgltf_skin* skin = &data->skins[0];
        cgltf_node* check = mesh_node->parent;
        while (check && parent_bone_idx < 0) {
          for (u32 j = 0; j < skin->joints_count; ++j) {
            if (skin->joints[j] == check) {
              parent_bone_idx = (int)j;
              break;
            }
          }
          check = check->parent;
        }
      }

      // Fix zero-weight vertices
      if (parent_bone_idx >= 0) {
        for (u32 v = 0; v < prim_vert_count; ++v) {
          u32 base = (vert_offset + v) * float_per_vert;
          f32 weight_sum = vdata[base + 12] + vdata[base + 13] + 
            vdata[base + 14] + vdata[base + 15];
          if (weight_sum < 0.001f) {
            vdata[base + 8] = (f32)parent_bone_idx;  // bone 0
            vdata[base + 12] = 1.0f;                  // weight 0
          }
        }
      }
      vert_offset += prim_vert_count;
 
    }
  }

  u32 vshape[ 2 ] = { total_verts, float_per_vert };
  tensor* t_verts = newTensor( 2, vshape, vdata );
  tensorToGPUMemory( t_verts );

  tensor* t_indices = NULL;
  if( total_indices > 0 ){
    u32 ishape[ 1 ] = { total_indices };
    t_indices = newTensor( 1, ishape, idata );
    tensorToGPUMemory( t_indices );
  }
  cgltf_free( data );
  unmem( fileData );

  
  f32* tp = mem( 1, f32 );
  *tp = animCount;
  tensor* t_animCount = newTensor( 0, NULL, tp );

  printf("Total verts: %d, indices: %d\n", total_verts, total_indices);
  // Return 4 Tensors: Verts, Indices, Anim, Textures
  tensor** result = mem( 5, tensor* );
  result[ 0 ] = t_verts;
  result[ 1 ] = t_indices;
  result[ 2 ] = t_anim;
  result[ 3 ] = t_tex; 
  result[ 4 ] = t_animCount; 

  if( outCount )
    *outCount = 5;
  return result;
}

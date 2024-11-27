import subprocess
import sys

def format_and_replace(filename):
    try:
        # Run clang-format -i on the file
        subprocess.run(["clang-format", "-i", filename], check=True)
        print(f"Formatted {filename} using clang-format.")

        # Read the file and perform the replacement
        with open(filename, "r") as file:
            content = file.read()
        updated_content = content.replace(") {", "){")

        # Write the updated content back to the file
        with open(filename, "w") as file:
            file.write(updated_content)
        print("Replaced")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running clang-format: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    format_and_replace(filename)

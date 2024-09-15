import os
import subprocess
import sys

# Function to create a virtual environment
def create_virtual_env(env_name):
    # Check if venv module is available
    if sys.version_info >= (3, 3):
        subprocess.check_call([sys.executable, "-m", "venv", env_name])
        print(f"Virtual environment '{env_name}' created.")
    else:
        print("venv module not available for this version of Python.")

# Function to make the activate script executable on Unix/MacOS
def make_activate_executable(env_name):
    activate_path = f"./{env_name}/bin/activate"
    if os.name != 'nt':  # For Unix/MacOS
        if os.path.exists(activate_path):
            os.chmod(activate_path, 0o755)  # Give execute permissions
            print(f"'{activate_path}' made executable.")
        else:
            print(f"'{activate_path}' does not exist.")

# Function to install libraries
def install_libraries(env_name, libraries):
    # Activate the virtual environment
    if os.name == 'nt':  # For Windows
        activate_env = f".\\{env_name}\\Scripts\\activate"
    else:  # For Unix/MacOS
        activate_env = f"./{env_name}/bin/activate"

    # Install the required libraries
    subprocess.call(f"{activate_env} && pip install {' '.join(libraries)}", shell=True)

if __name__ == "__main__":
    env_name = "my_env"  # Change environment name if needed
    libraries = ["numpy", "pandas", "scikit-learn", "mediapipe", "opencv-python", "tensorflow", "matplotlib"]  # Specify the libraries to install

    # Step 1: Create a virtual environment in the current directory
    # create_virtual_env(env_name)

    # Step 2: Make activate script executable (Unix/MacOS)
    make_activate_executable(env_name)

    # Step 3: Install the required libraries
    install_libraries(env_name, libraries)

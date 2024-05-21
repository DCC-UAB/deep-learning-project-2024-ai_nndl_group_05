import subprocess



def install_packages(requirements_file):
    with open(requirements_file, 'r') as f:
        requirements = f.readlines()
    
    # Strip newline characters
    requirements = [req.strip() for req in requirements if req.strip()]

    # Construct pip install command
    command = ['pip', 'install'] + requirements

    # Execute the command
    subprocess.call(command)


if __name__ == "__main__":
    requirements_file = 'requirements.txt'
    #install necessary packages 
    install_packages(requirements_file)
    
    

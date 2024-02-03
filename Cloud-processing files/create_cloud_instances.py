from google.cloud import storage
import subprocess
import sys

def create_instances(instance_type, count):
    command = ["gcloud", "compute", "instances", "create", "--count", str(count), "--machine-type", instance_type]
    subprocess.run(command)

def install_python(instance_name):
    command = ["gcloud", "compute", "ssh", instance_name, "--command", "sudo apt update && sudo apt install python3.11"]
    subprocess.run(command)

def install_libraries(instance_name):
    command = ["gcloud", "compute", "ssh", instance_name, "--command", "pip install mediapipe opencv-python tensorflow numpy google-cloud-storage"]
    subprocess.run(command)

def copy_python_file(instance_name, file_path):
    command = ["gcloud", "compute", "scp", file_path, instance_name:/home/user/]
    subprocess.run(command)

def copy_python_file(instance_name, file_path):
    command = ["gcloud", "compute", "scp", file_path, instance_name:/home/user/]
    subprocess.run(command)

def run_script(instance_name, script_path):
    command = ["gcloud", "compute", "ssh", instance_name, "--command", "python /home/user/" + script_path]
    subprocess.run(command)

def check_status(instance_name):
    command = ["gcloud", "compute", "ssh", instance_name, "--command", "echo 'Done'"]
    subprocess.run(command)

def shutdown(instance_name):
    command = ["gcloud", "compute", "instances", "stop", instance_name]
    subprocess.run(command)


# has to be run with 2 parameters:
#   1. the cloud storage adress
#   2. the amount of instances to be created
if __name__ == "__main__":
    instance_type = "n1-standard-1"
    count = sys.argv[2]
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(sys.argv[1])

    ALLE VIDEO-ADRESSEN BESORGEN

    video_adresses = [] # Enthält alle Adressen der Videos auf dem Blob Storage
    
    create_instances(instance_type, count)

    for i in range(1, count + 1):
        
        instance_name = "instance" + str(i)

        install_python(instance_name)
        install_libraries(instance_name)
        copy_python_file(instance_name, "cloud_pose_analyzer.py")
        run_script(instance_name, "cloud_pose_analyzer.py", video_adresses[i])
        check_status(instance_name)
        shutdown(instance_name)
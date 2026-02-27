import os

def list_structure(start_path, indent=""):
    try:
        items = sorted(os.listdir(start_path))
    except Exception as e:
        print(f"{indent}Error reading {start_path}: {e}")
        return

    for item in items:
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}/")
            list_structure(path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

print("Running tree check...\n")
print("Current working directory:", os.getcwd(), "\n")
print("Project Structure:\n")

list_structure(".")

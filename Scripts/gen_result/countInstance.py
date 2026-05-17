import os

# Path to your train.txt file
train_txt_path = "/home/ngerr/Workspace/Thesis/helmet_datasetv2_1/train.txt"

# Function to count class instances
def count_class_instances(train_txt_path):
    class_counts = {}

    with open(train_txt_path, 'r') as f:
        image_paths = f.readlines()

    for image_path in image_paths:
        image_path = image_path.strip()
        # Assuming label file is in the same directory as the image with a .txt extension
        label_path = image_path.replace(".jpg", ".txt").replace(".png", ".txt")
        
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found for {image_path}")
            continue
        
        with open(label_path, 'r') as label_file:
            for line in label_file:
                class_id = int(line.split()[0])  # First value in each line is the class ID
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts

# Run the function
class_counts = count_class_instances(train_txt_path)

# Display results
print("Class Instance Counts:")
for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count}")


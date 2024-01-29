import os
import PIL
import shutil

def validate_images (input_dir:str, output_dir:str, log_file:str, formatter:str = "07d") -> int:
    if not od.path.isdir(input_dir): #we check if the directory exists and raise a Value Error if it doesn't
        raise ValueError(f"Input directory '{input_dir}' does not exist!")

    os.makedirs(output_dir, exist_ok=True) #we create an output directory if the directory doesn't exist

    hash_set = set() #we initialize counts and hash set
    valid_count = 0
    invalid_count = 0

    with open(log_file, "w") as i: #this line opens the log file
        for root, dirs, files in walk(input_dir): #this line walks through the input directory
            files.sort() #this sorts the files in alphabetical order

            for file in files: #this gets the path of the file
                abs_path = os.path.join(root, file)

                if not file.lower().endswith((".jpg", ".jpeg")): #this block checks if the file name is valid
                    invalid_count += 1
                    i.write(f"{os.path.relpath(abs_path, input_dir)},1\n")
                    continue

                if os.path.getsize(abs_path) > 250000: #this block checks if the size is bigger than 250kb
                    invalid_count += 1
                    i.write(f"{os.path.relpath(abs_path, input_dir)},2\n")
                    continue

                try:                               #we try to check if the file can be read as an image
                    image = Image.open(abs_path)
                except:
                    invalid_count += 1
                    i.write(f"{os.path.relpath(abs_path, input_dir)},3\n")
                    continue

                if not (image.mode == "RGB" or image.mode == "L"):     #this block checks whether the file has the correct shape and mode
                    invalid_count += 1
                    i.write(f"{os.path.relpath(abs_path, input_dir)},4\n")
                    continue

                if len(image.size) != 2 or image.size[0] < 100 or image.size[1] < 100:
                    invalid_count += 1
                    i.write(f"{os.path.relpath(abs_path, input_dir)},4\n")
                    continue

                if image.getextrema()[0] == image.getextrema()[1]:           #this checks whether or not the image data has variance
                    invalid_count += 1
                    i.write(f"{os.path.relpath(abs_path, input_dir)},5\n")
                    continue

                file_hash = hash(image.tobytes())                        #this block checks whether or not the image has been copied
                if file_hash in hash_set:
                    invalid_count += 1
                    i.write(f"{os.path.relpath(abs_path, input_dir)},6\n")
                    continue

                base_name = f"{valid_count:{formatter}}.jpg"               #this block copies valid images to the output directory
                output_path = os.path.join(output_dir, base_name)
                shutil.copy(abs_path, output_path)

                valid_count += 1                 #we update the counts and hash set
                hash_set.add(file_hash)

    return valid_count             #this returns the number of valid files


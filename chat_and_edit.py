from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import json
import argparse
import traceback
from datetime import datetime
import uuid


from chat_interface import SketchApp
import utils  # Make sure to import utils module

app = Flask(__name__, static_folder='dist', static_url_path='/static')
CORS(app, resources={r"/*": {"origins": "*"}})  # More explicit CORS setup
build_dir = 'dist'
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if not path or path == '':
        return send_from_directory(build_dir, 'index.html')
    
    # Try to serve the file from the build directory
    file_path = os.path.join(build_dir, path)
    if os.path.isfile(file_path):
        directory, file = os.path.split(file_path)
        return send_from_directory(directory, file)
    
    # If not found and not an API route, serve index.html (for SPA routing)
    if not path.startswith('api/'):
        return send_from_directory(build_dir, 'index.html')
    
    return "Not Found", 404

@app.route('/static/sketches/<path:filename>')
def serve_sketch(filename):
    sketches_dir = 'static/sketches'
    return send_from_directory(sketches_dir, filename)
# Store current sketches in memory
sketches = {}

def create_args_for_concept(concept):
    """Create args object similar to what argparse would create"""
    args = argparse.Namespace()

    # General
    args.concept_to_draw = concept
    args.seed_mode = 'deterministic'

    # Create unique folder with absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.path2save = os.path.join(current_dir, f"results/api_{timestamp}_{session_id}")

    args.model = 'claude-3-5-sonnet-20240620'
    args.gen_mode = 'generation'

    # Grid params
    args.res = 50
    args.cell_size = 12
    args.stroke_width = 7.0
    args.grid_size = (args.res + 1) * args.cell_size

    args.save_name = args.concept_to_draw.replace(" ", "_")
    args.path2save = os.path.join(args.path2save, args.save_name)

    if not os.path.exists(args.path2save):
        os.makedirs(args.path2save)
        with open(os.path.join(args.path2save, "experiment_log.json"), 'w') as json_file:
            json.dump([], json_file, indent=4)

    return args

def find_matching_file(concept, directory='static/sketches'):
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Find files that contain the concept as a substring
    matching_files = [file for file in files if concept.lower() in file.lower() if file.lower().endswith('.png')]
    print(f"Matching files for '{concept}': {matching_files}")
    # Return the first matching file, if any
    if matching_files:
        return os.path.join(directory, matching_files[0])
    else:
        return None
    
@app.route('/generate-sketch', methods=['POST'])
def generate_sketch():
    try:
        data = request.get_json()
        concept = data.get('concept', '')
        if not concept:
            return jsonify({"error": "No concept provided"}), 400

        # Check if we have this sketch already
        # matching_file = find_matching_file(concept, 'static/sketches')
    
        # if matching_file:
        #     return jsonify({
        #         "message": f"Successfully found sketch of {concept}",
        #         "image_path": matching_file,
        #         "stroke_data": None
        #     })
        # Create args for SketchApp
        args = create_args_for_concept(concept)

        # Initialize SketchApp
        sketch_app = SketchApp(args)

        # Generate the sketch and get stroke data
        stroke_data = sketch_app.generate_sketch()

        # Get image path
        image_path = f"{args.path2save}/{args.save_name}.png"
        public_path = f"static/sketches/{args.save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # Copy to static folder for serving
        os.makedirs(os.path.dirname(f"static/sketches/"), exist_ok=True)

        # Use PIL to copy the image
        from PIL import Image
        img = Image.open(image_path)
        img.save(public_path)

        # Store information for later modifications
        sketches[concept] = {
            'original_path': image_path,
            'public_path': public_path,
            'args': args
        }

        return jsonify({
            "message": f"Successfully generated sketch of {concept}",
            "image_path": public_path,
            "stroke_data": stroke_data
        })

    except Exception as e:
        print(f"Error generating sketch: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/edit-sketch', methods=['POST'])
def edit_sketch():
    try:
        print("=== Starting edit-sketch endpoint ===")
        data = request.get_json()
        concept = data.get('concept', '')
        objects_to_add = data.get('objects_to_add', [])

        print(f"Request data: concept='{concept}', objects_to_add={objects_to_add}")

        if not concept or not objects_to_add:
            print("Missing required parameters")
            return jsonify({"error": "Both concept and objects_to_add must be provided"}), 400

        # Check if we have this sketch
        if concept not in sketches:
            print(f"Sketch '{concept}' not found in sketches dictionary")
            print(f"Available sketches: {list(sketches.keys())}")
            return jsonify({"error": f"No sketch found for '{concept}'"}), 404

        # Get original sketch info
        original_sketch_info = sketches[concept]
        print(f"Found sketch info: {original_sketch_info}")

        # Get the original image path - this is what we need to modify with
        original_image_path = original_sketch_info['original_path']
        print(f"Original image path: {original_image_path}")

        if not os.path.exists(original_image_path):
            print(f"WARNING: Original image does not exist at: {original_image_path}")
            return jsonify({"error": "Original image file not found"}), 500

        # Create a copy of the image for modification
        from PIL import Image
        original_image = Image.open(original_image_path)

        # Create a directory for modifications
        base_dir = os.path.dirname(original_image_path)
        modification_dir = os.path.join(base_dir, "modifications")
        os.makedirs(modification_dir, exist_ok=True)

        # Save a copy for the modification process
        mod_image_path = os.path.join(modification_dir, f"{concept}_canvas.png")
        original_image.save(mod_image_path)

        # Create output directory and copy experiment_log.json
        exp_log_path = os.path.join(base_dir, "experiment_log.json")
        if os.path.exists(exp_log_path):
            with open(exp_log_path, 'r') as f:
                experiment_log = json.load(f)

            with open(os.path.join(modification_dir, "experiment_log.json"), 'w') as f:
                json.dump(experiment_log, f, indent=4)

        # Use the SketchApp class method to edit the sketch
        sketch_app = SketchApp(original_sketch_info['args'])
        print(f"Created SketchApp instance")

        # Define reflection prompt
        reflection_prompt = "Please add {add_object} to the existing sketch of {object_to_edit}."

        # Prepare the edit_sketch_in_chat_add parameters
        path_to_data = os.path.dirname(base_dir)

        # Create a simple XML file that just contains the basic sketch
        # This simulates what load_sketch_data expects
        concept = concept.replace(" ", "_")  # Ensure no spaces in concept name
        output_filename = f"output_{concept}_canvas.png"
        output_path = os.path.join(path_to_data, concept)
        os.makedirs(output_path, exist_ok=True)

        # Copy the original image to where load_sketch_data expects it
        original_image.save(os.path.join(output_path, output_filename))

        print(f"Set up temporary file at: {os.path.join(output_path, output_filename)}")

        # Now call the edit method
        results = sketch_app.edit_sketch_in_chat_add(
            path_to_data=path_to_data,
            object_to_edit=concept,
            add_objects=objects_to_add,
            reflection_prompt=reflection_prompt,
            cache=False,
            seed_mode="deterministic"
        )

        print(f"Edit sketch results keys: {results.keys() if results else 'None'}")

        # Get the final image and stroke data from the results
        final_image = results.get("final_image")
        stroke_data = results.get("stroke_data")

        if final_image is None:
            print("No final_image in results")
            return jsonify({"error": "Failed to generate edited sketch"}), 500

        # Generate a path for the edited image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        edited_concept = f"{concept} with {', '.join(objects_to_add)}"
        edited_name = edited_concept.replace(" ", "_")

        # Create a path in the static folder
        os.makedirs("static/sketches", exist_ok=True)
        public_path = f"static/sketches/{edited_name}_{timestamp}.png"
        full_public_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), public_path)

        print(f"Will save edited image to {full_public_path}")

        # Save the image
        final_image.save(full_public_path)
        print(f"Image saved")

        # Store the edited sketch info
        sketches[edited_concept] = {
            'original_path': full_public_path,
            'public_path': public_path,
            'args': original_sketch_info['args'],
            'parent_concept': concept
        }

        print("=== Successfully completed edit-sketch endpoint ===")
        return jsonify({
            "message": f"Successfully added {', '.join(objects_to_add)} to sketch of {concept}",
            "image_path": public_path,
            "stroke_data": stroke_data
        })

    except Exception as e:
        print(f"=== ERROR in edit-sketch endpoint: {e} ===")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
from pathlib import Path
import socket

def main() -> None:
    # 1. Make sure the folder for generated sketches exists
    Path("static/sketches").mkdir(parents=True, exist_ok=True)

    PORT  = 5000
    HOST  = "0.0.0.0"          # listen on *every* interface so remote traffic can reach us
    try:
        # The “dummy UDP” trick: no packets leave the box, but we learn
        # which interface would be used to reach the Internet.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
    except OSError:
        ip_address = "127.0.0.1"

    # 3. Helpful message for *both* access patterns
    print(
        f"\n📡  SketchApp server running!\n"
        f"    • Direct:  http://{ip_address}:{PORT}\n"
        f"    • Via SSH: ssh -L {PORT}:localhost:{PORT} <user>@{ip_address}\n"
        f"      then open  →  http://localhost:{PORT}\n"
    )

    # 4. Start Flask (blocking call)
    app.run(host=HOST, port=PORT, debug=True)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()




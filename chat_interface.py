from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import argparse
import anthropic
import ast
import cairosvg
import json
import utils
import traceback
from datetime import datetime
import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from prompts import sketch_first_prompt, system_prompt, gt_example

app = Flask(__name__)  # This line defines the app
CORS(app)


def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')

    # General
    parser.add_argument('--concept_to_draw', type=str, default="cat")
    parser.add_argument('--seed_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])
    parser.add_argument('--path2save', type=str, default=f"results/test")
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20240620')
    parser.add_argument('--gen_mode', type=str, default='generation', choices=['generation', 'completion'])

    # Grid params
    parser.add_argument('--res', type=int, default=50, help="the resolution of the grid is set to 50x50")
    parser.add_argument('--cell_size', type=int, default=12, help="size of each cell in the grid")
    parser.add_argument('--stroke_width', type=float, default=7.0)

    args = parser.parse_args()
    args.grid_size = (args.res + 1) * args.cell_size

    args.save_name = args.concept_to_draw.replace(" ", "_")
    args.path2save = f"{args.path2save}/{args.save_name}"
    if not os.path.exists(args.path2save):
        os.makedirs(args.path2save)
        with open(f"{args.path2save}/experiment_log.json", 'w') as json_file:
            json.dump([], json_file, indent=4)
    return args


sketches = {}


def create_args_for_concept(concept):
    """Create args object similar to what argparse would create"""
    from datetime import datetime
    import uuid

    args = argparse.Namespace()

    # General
    args.concept_to_draw = concept
    args.seed_mode = 'deterministic'

    # Create unique folder for each session
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.path2save = f"results/api_{timestamp}_{session_id}"

    args.model = 'claude-3-5-sonnet-20240620'
    args.gen_mode = 'generation'

    # Grid params
    args.res = 50
    args.cell_size = 12
    args.stroke_width = 7.0
    args.grid_size = (args.res + 1) * args.cell_size

    args.save_name = args.concept_to_draw.replace(" ", "_")
    args.path2save = f"{args.path2save}/{args.save_name}"

    if not os.path.exists(args.path2save):
        os.makedirs(args.path2save)
        with open(f"{args.path2save}/experiment_log.json", 'w') as json_file:
            json.dump([], json_file, indent=4)

    return args

def load_sketch_data(path_to_data: str, object_to_edit: str, cache: bool = False):
    """
    Returns
    -------
    sketch_rendered : PIL.Image.Image
    system_prompt   : str | list
    msg_history     : list
    assistant_prompt: str
    """
    # 1️⃣  normalise the directory name exactly the same way you did when saving
    object_dir_name = object_to_edit.replace(" ", "_")   # update to your own slug-rule
    obj_dir = Path(path_to_data) / object_dir_name

    im_path  = obj_dir / f"output_{object_dir_name}_canvas.png"
    json_path = obj_dir / "experiment_log.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found.\n"
            f"Available objects in {path_to_data}: "
            f"{[p.name for p in Path(path_to_data).iterdir()]}"
        )

    with json_path.open() as f:
        log = json.load(f)

    system_prompt = log[0]["content"][0]["text"] if cache else log[0]["content"]
    assistant_prompt = log[-1]["content"][0]["text"]
    msg_history = log[1:]

    sketch_rendered = Image.open(im_path)
    return sketch_rendered, system_prompt, msg_history, assistant_prompt

def call_llm(system_message, other_msg, cache, additional_args):
    if cache:
        init_response = client.beta.prompt_caching.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_message,
                messages=other_msg,
                **additional_args
            )
    else:
        init_response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_message,
                messages=other_msg,
                **additional_args
            )
    return init_response

def define_input_to_llm(msg_history, init_canvas_str, msg, cache):
    # other_msg should contain all messgae without the system prompt
    other_msg = msg_history

    content = []
    # Claude best practice is image-then-text
    if init_canvas_str is not None:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": init_canvas_str}})

    content.append({"type": "text", "text": msg})
    if cache:
        content[-1]["cache_control"] = {"type": "ephemeral"}

    other_msg = other_msg + [{"role": "user", "content": content}]
    return other_msg

def get_response_from_llm(
        msg,
        system_message,
        msg_history=[],
        init_canvas_str=None,
        prefill_msg=None,
        seed_mode="stochastic",
        stop_sequences=None,
        gen_mode="generation",
        cache=True,
        path2save=None
    ):
        additional_args = {}
        if seed_mode == "deterministic":
            additional_args["temperature"] = 0.0
            additional_args["top_k"] = 1

        if cache:
            system_message = [{
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"}
            }]

        # other_msg should contain all messgae without the system prompt
        other_msg = define_input_to_llm(msg_history, init_canvas_str, msg, cache)

        if gen_mode == "completion":
            if prefill_msg:
                other_msg = other_msg + [{"role": "assistant", "content": f"{prefill_msg}"}]

            # in case of stroke by stroke generation
        if stop_sequences:
            additional_args["stop_sequences"]= [stop_sequences]
        else:
            additional_args["stop_sequences"]= ["</answer>"]

        # Note that we deterministic settings for reproducibility (temperature=0.0 and top_k=1).
        # To run in stochastic mode just comment these parameters.
        response = call_llm(system_message, other_msg, cache, additional_args)

        content = response.content[0].text

        if gen_mode == "completion":
            other_msg = other_msg[:-1] # remove initial assistant prompt
            content = f"{prefill_msg}{content}"

        # saves to json
        if path2save is not None:
            system_message_json = [{"role": "system", "content": system_message}]
            new_msg_history = other_msg + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            ]
            with open(f"{path2save}/experiment_log.json", 'w') as json_file:
                json.dump(system_message_json + new_msg_history, json_file, indent=4)
            print(f"Data has been saved to [{path2save}/experiment_log.json]")

        return content, new_msg_history


def save_sketch(model_strokes_svg, output_path, add_object, init_canvas):
    with open(f"{output_path}/output_{add_object}.svg", "w") as svg_file:
        svg_file.write(model_strokes_svg)

    # Save the result with clean white background (no grid)
    cairosvg.svg2png(url=f"{output_path}/output_{add_object}.svg",
                     write_to=f"{output_path}/output_{add_object}.png",
                     background_color="white")

    if init_canvas is not None:
        # For the canvas version, create a blank white canvas instead of using the grid
        output_png_path = f"{output_path}/output_{add_object}_canvas.png"

        # Create a blank white image with the same dimensions as init_canvas
        blank_canvas = Image.new('RGB', init_canvas.size, 'white')

        # Convert SVG to PNG and overlay on blank canvas
        cairosvg.svg2png(url=f"{output_path}/output_{add_object}.svg", write_to=output_png_path)
        foreground = Image.open(output_png_path)
        blank_canvas.paste(foreground, (0, 0), foreground)
        blank_canvas.save(output_png_path)

        return blank_canvas

    # # save the result also without the canvas background
    # cairosvg.svg2png(url=f"{output_path}/output_{add_object}.svg", write_to=f"{output_path}/output_{add_object}.png", background_color="white")

    # if init_canvas is not None:
    #     # save the result as png on the canvas background
    #     output_png_path = f"{output_path}/output_{add_object}_canvas.png"
    #     cairosvg.svg2png(url=f"{output_path}/output_{add_object}.svg", write_to=output_png_path)
    #     foreground = Image.open(output_png_path)
    #     init_canvas_copy = init_canvas.copy()
    #     init_canvas_copy.paste(Image.open(output_png_path), (0, 0), foreground)
    #     init_canvas_copy.save(output_png_path)
    #     return init_canvas_copy

class SketchApp:
    def __init__(self, args):
        # General
        self.path2save = args.path2save
        self.target_concept = args.concept_to_draw
        self.save_name = args.save_name if hasattr(args, 'save_name') else self.target_concept.replace(" ", "_")
        self.args = args

        # Grid related
        self.res = args.res
        self.num_cells = args.res
        self.cell_size = args.cell_size
        self.grid_size = (args.grid_size, args.grid_size)
        self.init_canvas, self.positions = utils.create_grid_image(res=args.res, cell_size=args.cell_size, header_size=args.cell_size)
        self.init_canvas_str = utils.image_to_str(self.init_canvas)
        self.cells_to_pixels_map = utils.cells_to_pixels(args.res, args.cell_size, header_size=args.cell_size)

        # SVG related
        self.stroke_width = args.stroke_width

        # LLM Setup (you need to provide your ANTHROPIC_API_KEY in your .env file)
        self.cache = False
        self.max_tokens = 3000
        load_dotenv()
        claude_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=claude_key)
        self.model = args.model
        self.input_prompt = sketch_first_prompt.format(concept=args.concept_to_draw, gt_sketches_str=gt_example)
        self.gen_mode = args.gen_mode
        self.seed_mode = args.seed_mode

    def call_llm(self, system_message, other_msg, additional_args):
        if self.cache:
            init_response = self.client.beta.prompt_caching.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_message,
                messages=other_msg,
                **additional_args
            )
        else:
            init_response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_message,
                messages=other_msg,
                **additional_args
            )
        return init_response

    def define_input_to_llm(self, msg_history, init_canvas_str, msg):
        # other_msg should contain all messgae without the system prompt
        other_msg = msg_history

        content = []
        # Claude best practice is image-then-text
        if init_canvas_str is not None:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": init_canvas_str}})

        content.append({"type": "text", "text": msg})
        if self.cache:
            content[-1]["cache_control"] = {"type": "ephemeral"}

        other_msg = other_msg + [{"role": "user", "content": content}]
        return other_msg

    def get_response_from_llm(
        self,
        msg,
        system_message,
        msg_history=[],
        init_canvas_str=None,
        prefill_msg=None,
        seed_mode="stochastic",
        stop_sequences=None,
        gen_mode="generation"
    ):
        additional_args = {}
        if seed_mode == "deterministic":
            additional_args["temperature"] = 0.0
            additional_args["top_k"] = 1

        if self.cache:
            system_message = [{
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"}
            }]

        # other_msg should contain all messgae without the system prompt
        other_msg = self.define_input_to_llm(msg_history, init_canvas_str, msg)

        if gen_mode == "completion":
            if prefill_msg:
                other_msg = other_msg + [{"role": "assistant", "content": f"{prefill_msg}"}]

        # In case of stroke by stroke generation
        if stop_sequences:
            additional_args["stop_sequences"] = [stop_sequences]
        else:
            additional_args["stop_sequences"] = ["</answer>"]

        response = self.call_llm(system_message, other_msg, additional_args)
        content = response.content[0].text

        if gen_mode == "completion":
            other_msg = other_msg[:-1]  # remove initial assistant prompt
            content = f"{prefill_msg}{content}"

        # saves to json
        if self.path2save is not None:
            system_message_json = [{"role": "system", "content": system_message}]
            new_msg_history = other_msg + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            ]
            with open(f"{self.path2save}/experiment_log.json", 'w') as json_file:
                json.dump(system_message_json + new_msg_history, json_file, indent=4)
            print(f"Data has been saved to [{self.path2save}/experiment_log.json]")
            print(content)
        return content

    def call_model_for_sketch_generation(self):
        print("Calling LLM for sketch generation...")
        print("Input Prompt:", self.input_prompt)

        add_args = {}
        add_args["stop_sequences"] = f"</answer>"

        msg_history = []
        init_canvas_str = None  # self.init_canvas_str

        try:
            all_llm_output = self.get_response_from_llm(
                msg=self.input_prompt,
                system_message=system_prompt.format(res=self.res),
                msg_history=msg_history,
                init_canvas_str=init_canvas_str,
                seed_mode=self.seed_mode,
                gen_mode=self.gen_mode,
                **add_args
            )

            all_llm_output += f"</answer>"

            print("Full LLM Output:")
            print(all_llm_output)

            return all_llm_output

        except Exception as e:
            print(f"Error in LLM call: {e}")
            import traceback
            traceback.print_exc()

            # Return a fallback output
            return self.get_default_stroke_data()

    def parse_model_to_svg(self, model_rep_sketch):
        # Parse model_rep with xml
        strokes_list_str, t_values_str = utils.parse_xml_string(model_rep_sketch, self.res)
        strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)

        # extract control points from sampled lists
        all_control_points = utils.get_control_points(strokes_list, t_values, self.cells_to_pixels_map)

        # define SVG based on control point
        sketch_text_svg = utils.format_svg(all_control_points, dim=self.grid_size, stroke_width=self.stroke_width)
        return sketch_text_svg

    def generate_sketch(self):
        # Call the LLM to get sketching commands
        sketching_commands = self.call_model_for_sketch_generation()

        # Parse the commands to get strokes
        model_strokes_svg = self.parse_model_to_svg(sketching_commands)

        # Save the SVG sketch
        svg_path = f"{self.path2save}/{self.save_name}.svg"
        with open(svg_path, "w") as svg_file:
            svg_file.write(model_strokes_svg)

        # Convert to PNG
        png_path = f"{self.path2save}/{self.save_name}.png"
        cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="white")

        # Save with canvas background
        canvas_png_path = f"{self.path2save}/{self.save_name}_canvas.png"
        cairosvg.svg2png(url=svg_path, write_to=canvas_png_path)
        foreground = Image.open(canvas_png_path)
        self.init_canvas.paste(Image.open(canvas_png_path), (0, 0), foreground)
        self.init_canvas.save(canvas_png_path)

        # Generate stroke data in XML format
        stroke_data = self.extract_stroke_data_from_llm_output(sketching_commands)
        return stroke_data

    def extract_stroke_data_from_llm_output(self, llm_output):
        """Extract stroke data from LLM output and format as XML."""
        try:
            # Aggressive debugging - print EXACTLY what was received
            print("RAW LLM OUTPUT (START)")
            print(repr(llm_output))
            print("RAW LLM OUTPUT (END)")

            # Create root XML element
            root = ET.Element("answer")

            # Add concept
            concept_elem = ET.SubElement(root, "concept")
            concept_elem.text = self.target_concept

            # Create strokes element
            strokes_elem = ET.SubElement(root, "strokes")

            import re

            # Pattern to match entire stroke sections
            stroke_pattern = r'<s(\d+)>\n*\s*<points>(.*?)</points>\n*\s*<t_values>(.*?)</t_values>\n*\s*<id>(.*?)</id>\n*\s*</s\1>'

            # Find all stroke matches
            stroke_matches = re.findall(stroke_pattern, llm_output, re.DOTALL | re.IGNORECASE)

            stroke_count = 0
            for match in stroke_matches:
                stroke_num, points, t_values, stroke_id = match

                # Clean and process points
                point_pattern = r"'(x\d+y\d+)'"
                parsed_points = re.findall(point_pattern, points)

                # Clean and process t-values
                parsed_t_values = [val.strip() for val in t_values.split(',')]

                # Create stroke element
                stroke_elem = ET.SubElement(strokes_elem, f"s{stroke_num}")

                # Add points
                points_elem = ET.SubElement(stroke_elem, "points")
                points_elem.text = ", ".join([f"'{p}'" for p in parsed_points])

                # Add t-values
                t_values_elem = ET.SubElement(stroke_elem, "t_values")
                t_values_elem.text = ", ".join(parsed_t_values)

                # Add ID
                id_elem = ET.SubElement(stroke_elem, "id")
                id_elem.text = stroke_id.strip()

                stroke_count += 1

            # If no strokes found, fallback to default
            if stroke_count == 0:
                print("No strokes found. Falling back to default stroke.")
                stroke_elem = ET.SubElement(strokes_elem, "s1")

                points_elem = ET.SubElement(stroke_elem, "points")
                points_elem.text = "'x10y10', 'x40y10', 'x40y40', 'x10y40', 'x10y10'"

                t_values_elem = ET.SubElement(stroke_elem, "t_values")
                t_values_elem.text = "0.00, 0.25, 0.50, 0.75, 1.00"

                id_elem = ET.SubElement(stroke_elem, "id")
                id_elem.text = "fallback_stroke"

            # Print the number of strokes found for debugging
            print(f"Number of strokes extracted: {stroke_count}")

            # Format the XML nicely
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            print("Generated Stroke XML:", xml_str)  # Add this for debugging
            return xml_str

        except Exception as e:
            print(f"Error extracting stroke data: {e}")
            traceback.print_exc()

            # Return a simple fallback
            return self.get_default_stroke_data()

    def get_default_stroke_data(self):
        """Generate default stroke data if extraction fails."""
        root = ET.Element("answer")

        concept_elem = ET.SubElement(root, "concept")
        concept_elem.text = self.target_concept

        strokes_elem = ET.SubElement(root, "strokes")

        # Add a simple placeholder stroke
        stroke1 = ET.SubElement(strokes_elem, "s1")
        points1 = ET.SubElement(stroke1, "points")
        points1.text = "'x10y10', 'x40y10', 'x40y40', 'x10y40', 'x10y10'"
        t_values1 = ET.SubElement(stroke1, "t_values")
        t_values1.text = "0.00, 0.25, 0.50, 0.75, 1.00"
        id1 = ET.SubElement(stroke1, "id")
        id1.text = "outline"

        # Format the XML nicely
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        return xml_str

    def get_stroke_data(self):
        """Return the stroke data in XML format"""
        # This is where you would generate or retrieve the stroke data
        # For this example, I'm creating a simple structure based on your XML
        # In your actual implementation, this would come from your sketch generation logic

        root = ET.Element("answer")

        concept = ET.SubElement(root, "concept")
        concept.text = self.args.concept_to_draw

        strokes = ET.SubElement(root, "strokes")

        # Example of adding a stroke - you'll populate this with your actual stroke data
        # This is just a placeholder based on your example
        stroke1 = ET.SubElement(strokes, "s1")
        points1 = ET.SubElement(stroke1, "points")
        points1.text = "'x10y25', 'x40y25', 'x40y25', 'x40y10', 'x40y10', 'x10y10', 'x10y10', 'x10y25'"
        t_values1 = ET.SubElement(stroke1, "t_values")
        t_values1.text = "0.00,0.25,0.25,0.50,0.50,0.75,0.75,1.00"
        id1 = ET.SubElement(stroke1, "id")
        id1.text = "building base"

        # Format the XML nicely
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        return xml_str


    def edit_sketch_in_chat_add(self, path_to_data, object_to_edit, add_objects, reflection_prompt, cache=True, seed_mode="deterministic"):
        """
        Method to edit an existing sketch by adding new objects incrementally.
        Each object is added separately and strokes are accumulated.
        """
        object_to_edit = object_to_edit.replace(" ", "_")  # Normalise the object name
        output_path = f"{path_to_data}/{object_to_edit}/editing_add"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Load sketch data
        sketch_rendered, system_prompt, msg_history, assistant_prompt = load_sketch_data(path_to_data, object_to_edit, cache)
        with open(f"{output_path}/experiment_log.json", 'w') as json_file:
            system_message_json = [{"role": "system", "content": system_prompt}]
            json.dump(system_message_json + msg_history, json_file, indent=4)

        # Save given strokes
        prev_strokes_list_str, prev_t_values_str = utils.parse_xml_string(assistant_prompt, res=self.res)
        accum_strokes_list, accum_t_values = ast.literal_eval(prev_strokes_list_str), ast.literal_eval(prev_t_values_str)
        cur_sketch_str = utils.image_to_str(sketch_rendered)

        # Add objects in a loop
        for add_object in add_objects:
            user_edit_prompt = reflection_prompt.format(add_object=add_object, object_to_edit=object_to_edit)

            all_llm_output = self.get_response_from_llm(
                        msg=user_edit_prompt,
                        system_message=system_prompt,
                        msg_history=msg_history,
                        init_canvas_str=cur_sketch_str,
                        seed_mode=seed_mode,
                        gen_mode=self.gen_mode
                    )

            strokes_list_str, t_values_str = utils.parse_xml_string(all_llm_output, res=self.res)
            strokes_list, t_values = ast.literal_eval(strokes_list_str), ast.literal_eval(t_values_str)

            # This is the part where we add the new strokes to existing ones:
            accum_strokes_list.extend(strokes_list)
            accum_t_values.extend(t_values)
            all_control_points = utils.get_control_points(accum_strokes_list, accum_t_values, self.cells_to_pixels_map)
            model_strokes_svg = utils.format_svg(all_control_points, dim=self.grid_size, stroke_width=self.stroke_width)
            sketch_rendered = save_sketch(model_strokes_svg, output_path, add_object, self.init_canvas)

            cur_sketch_str = utils.image_to_str(sketch_rendered)
            msg_history = msg_history + [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": all_llm_output,
                            }
                        ],
                    }
                ]

        # Return final results including the new strokes
        return {
            "final_image": sketch_rendered,
            "stroke_data": self.format_stroke_data_for_frontend(accum_strokes_list, accum_t_values, object_to_edit, add_objects)
        }

    def format_stroke_data_for_frontend(self, strokes_list, t_values, original_concept, added_objects):
        """Format stroke data in XML format for frontend animation"""
        root = ET.Element("answer")

        # Add concept
        concept_elem = ET.SubElement(root, "concept")
        concept_elem.text = f"{original_concept} with {', '.join(added_objects)}"

        # Create strokes element
        strokes_elem = ET.SubElement(root, "strokes")

        # Add each stroke to the XML
        for i, (points, t_vals) in enumerate(zip(strokes_list, t_values)):
            stroke_elem = ET.SubElement(strokes_elem, f"s{i+1}")

            # Add points
            points_elem = ET.SubElement(stroke_elem, "points")
            points_elem.text = ", ".join([f"'{p}'" for p in points])

            # Add t-values
            t_values_elem = ET.SubElement(stroke_elem, "t_values")
            t_values_elem.text = ", ".join([str(t) for t in t_vals])

            # Add ID
            id_elem = ET.SubElement(stroke_elem, "id")
            id_elem.text = f"stroke_{i+1}"

        # Format the XML nicely
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        return xml_str




# Initialize and run the SketchApp
if __name__ == '__main__':
    args = call_argparse()
    sketch_app = SketchApp(args)
    for attempts in range(3):
        try:
            sketch_app.generate_sketch()
            exit(0)
        except Exception as e:
            print(f"An error has occurred: {e}")
            traceback.print_exc()

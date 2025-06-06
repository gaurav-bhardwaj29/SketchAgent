import argparse
from openai import OpenAI
import ast
import cairosvg
import json
import os
import utils
import traceback

from dotenv import load_dotenv
from PIL import Image
from prompts import sketch_first_prompt, system_prompt, gt_example


def call_argparse():
    parser = argparse.ArgumentParser(description='Process Arguments')
    
    # General
    parser.add_argument('--concept_to_draw', type=str, default="cat")
    parser.add_argument('--seed_mode', type=str, default='deterministic', choices=['deterministic', 'stochastic'])
    parser.add_argument('--path2save', type=str, default=f"results/test")
    parser.add_argument('--model', type=str, default='gpt-4o')
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


class SketchApp:
    def __init__(self, args):
        # General
        self.path2save = args.path2save
        self.target_concept = args.concept_to_draw

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
        
        # LLM Setup (you need to provide your OPENAI_API_KEY in your .env file)
        # self.cache = False
        self.max_tokens = 3000
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=openai_key)
        self.model = "gpt-4o"
        self.input_prompt = sketch_first_prompt.format(concept=args.concept_to_draw, gt_sketches_str=gt_example)
        self.gen_mode = args.gen_mode
        self.seed_mode = args.seed_mode
        

    def call_llm(self, system_message, other_msg, additional_args):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_message}] + other_msg,
            max_tokens=self.max_tokens,
            temperature=additional_args.get("temperature", 0.0),
            stop=additional_args.get("stop", None)
        )
        return response.choices[0].message.content

    
    def define_input_to_llm(self, msg_history, init_canvas_str, msg):
        content = []
        if init_canvas_str is not None:
            content.append({
                "type": "image_url",
                "image_url": "data:image/jpeg;base64," + init_canvas_str
            })
        content.append({"type": "text", "text": msg})

        other_msg = msg_history + [{"role": "user", "content": content}]
        return other_msg
        

    def get_response_from_llm(
        self,
        msg,
        system_message,
        msg_history=[],
        init_canvas_str=None,
        prefill_msg=None,
        # seed_mode="stochastic",
        stop=None,
        gen_mode="generation"
    ):  
        additional_args = {}
        # if seed_mode == "deterministic":
            #additional_args["temperature"] = 0.0
            #additional_args["top_k"] = 1

        # if self.cache:
        #     system_message = [{
        #         "type": "text",
        #         "text": system_message,
        #         "cache_control": {"type": "ephemeral"}
        #     }]

        # other_msg should contain all messgae without the system prompt
        other_msg = self.define_input_to_llm(msg_history, init_canvas_str, msg) 

        if gen_mode == "completion":
            if prefill_msg:
                other_msg = other_msg + [{"role": "assistant", "content": f"{prefill_msg}"}]
            
        # In case of stroke by stroke generation
        if stop:
            additional_args["stop"]= stop
        else:
            additional_args["stop"]= ["</answer>"]
 
        response = self.call_llm(system_message, other_msg, additional_args)
        content = response
        
        if gen_mode == "completion":
            other_msg = other_msg[:-1] # remove initial assistant prompt
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
        return content


    def call_model_for_sketch_generation(self):
        print("Calling LLM...")
        
        add_args = {}
        add_args["stop"] = "</answer>" 

        msg_history = []
        init_canvas_str = None # self.init_canvas_str

        all_llm_output = self.get_response_from_llm(
            msg=self.input_prompt,
            system_message=system_prompt.format(res=self.res),
            msg_history=msg_history,
            init_canvas_str=init_canvas_str,
            #seed_mode=self.seed_mode,
            gen_mode=self.gen_mode,
            **add_args
        )

        all_llm_output += f"</answer>"
        return all_llm_output
        

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
        sketching_commands = self.call_model_for_sketch_generation()
        model_strokes_svg = self.parse_model_to_svg(sketching_commands)
        # saved the SVG sketch
        with open(f"{self.path2save}/{self.target_concept}.svg", "w") as svg_file:
            svg_file.write(model_strokes_svg)

        # vector->pixel 
        # save the sketch to png with blank backgournd
        cairosvg.svg2png(url=f"{self.path2save}/{self.target_concept}.svg", write_to=f"{self.path2save}/{self.target_concept}.png", background_color="white")
        
        # save the sketch to png on the canvas
        output_png_path = f"{self.path2save}/{self.target_concept}_canvas.png"
        cairosvg.svg2png(url=f"{self.path2save}/{self.target_concept}.svg", write_to=output_png_path)
        foreground = Image.open(output_png_path)
        self.init_canvas.paste(Image.open(output_png_path), (0, 0), foreground) 
        self.init_canvas.save(output_png_path)

        

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
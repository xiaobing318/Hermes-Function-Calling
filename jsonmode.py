# 导入第三方库argparse，这个库的作用就是管理命令行参数相关信息的
import argparse
# 导入第三方库torch，这个库就是用来train and inference一些model
import torch
# 导入第三方库json，这个库的作用就是为了处理有关json数据相关内容的
import json
# 导入第三方库transformers中的一些类，这些类和model inference是相关的
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
# 导入第三方库validator中的validate_json_data函数
from validator import validate_json_data
# 导入第三方库utils中导入多个函数
from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)
# 导入第三方库typing中的List、Optional两个类
# create your pydantic model for json object here
from typing import List, Optional
# 导入第三方库pydantic中的BaseModel
from pydantic import BaseModel

# 声明一个名为Character的、继承自BaseModel的类
class Character(BaseModel):
    # 类的成员变量，成员变量的类型是string
    name: str
    # 类的成员变量，成员变量的类型是string
    species: str
    # 类的成员变量，成员变量的类型是string
    role: str
    # 类的成员变量
    personality_traits: Optional[List[str]]
    # 类的成员变量
    special_attacks: Optional[List[str]]
    # 声明一个名为Config的类
    class Config:
        schema_extra = {
            "additionalProperties": False
        }

# serialize pydantic model into json schema
pydantic_schema = Character.schema_json()

# 声明一个名为ModelInference的类
class ModelInference:
    # 类的构造函数
    def __init__(self, model_path, chat_template, load_in_4bit):
        inference_logger.info(print_nous_text_art())
        self.bnb_config = None

        if load_in_4bit == "True":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            return_dict=True,
            quantization_config=self.bnb_config,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.tokenizer.chat_template is None:
            print("No chat template defined, getting chat_template...")
            self.tokenizer.chat_template = get_chat_template(chat_template)
        
        inference_logger.info(self.model.config)
        inference_logger.info(self.model.generation_config)
        inference_logger.info(self.tokenizer.special_tokens_map)
    # 类的成员函数
    def run_inference(self, prompt):
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=1500,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        completion = self.tokenizer.decode(tokens[0], skip_special_tokens=False, clean_up_tokenization_space=True)
        return completion
    # 类的成员函数
    def generate_json_completion(self, query, chat_template, max_depth=5):
        try:
            depth = 0
            sys_prompt = f"You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{pydantic_schema}\n</schema>"
            prompt = [{"role": "system", "content": sys_prompt}]
            prompt.append({"role": "user", "content": query})

            inference_logger.info(f"Running inference to generate json object for pydantic schema:\n{json.dumps(json.loads(pydantic_schema), indent=2)}")
            completion = self.run_inference(prompt)

            def recursive_loop(prompt, completion, depth):
                nonlocal max_depth

                assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)

                tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
                if assistant_message is not None:
                    validation, json_object, error_message = validate_json_data(assistant_message, json.loads(pydantic_schema))
                    if validation:
                        inference_logger.info(f"Assistant Message:\n{assistant_message}")
                        inference_logger.info(f"json schema validation passed")
                        inference_logger.info(f"parsed json object:\n{json.dumps(json_object, indent=2)}")
                    elif error_message:
                        inference_logger.info(f"Assistant Message:\n{assistant_message}")
                        inference_logger.info(f"json schema validation failed")
                        tool_message += f"<tool_response>\nJson schema validation failed\nHere's the error stacktrace: {error_message}\nPlease return corrrect json object\n<tool_response>"
                        
                        depth += 1
                        if depth >= max_depth:
                            print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                            return
                        
                        prompt.append({"role": "tool", "content": tool_message})
                        completion = self.run_inference(prompt)
                        recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.warning("Assistant message is None")
            recursive_loop(prompt, completion, depth)
        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e

# 这行代码用于检查当前模块是否为主程序运行环境。只有当文件作为主程序执行时（不是作为模块被导入到其他文件中），代码块内的内容才会执行。
if __name__ == "__main__":
    # 这行代码创建了一个 `ArgumentParser` 对象，该对象用于解析命令行参数。`description` 参数提供了这个程序的简短描述。
    parser = argparse.ArgumentParser(description="Run json mode completion")
    # 向解析器添加一个命令行参数 `--model_path`，这是一个字符串类型的参数，用于指定模型文件的路径。
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    # 添加一个 `--chat_template` 参数，它也是字符串类型，有一个默认值 `"chatml"`。此参数用于指定聊天提示的格式模板。
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    # 添加 `--load_in_4bit` 参数，它是一个字符串类型的参数，用于决定是否使用 `bitsandbytes` 库将模型加载为 4bit 格式。默认为 `"False"`。
    parser.add_argument("--load_in_4bit", type=str, default="False", help="Option to load in 4bit with bitsandbytes")
    # 添加 `--query` 参数，它是一个字符串类型，具有默认值，用于定义需要处理的默认查询。
    parser.add_argument("--query", type=str, default="Please return a json object to represent Goku from the anime Dragon Ball Z?")
    # 添加 `--max_depth` 参数，这是一个整数类型的参数，用来设置递归调用的最大深度，默认为 5。
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of recursive iteration")
    # 这行代码解析命令行输入的参数，并将解析后的参数存储在 `args` 对象中。
    args = parser.parse_args()

    # specify custom model path(如果 `model_path` 存在，使用提供的参数创建一个 `ModelInference` 对象)
    if args.model_path:
        inference = ModelInference(args.model_path, args.chat_template, args.load_in_4bit)
    else:
        # 设置默认的模型路径。
        model_path = 'NousResearch/Hermes-2-Pro-Llama-3-8B'
        # 使用默认的模型路径创建一个新的 `ModelInference` 对象。
        inference = ModelInference(model_path, args.chat_template, args.load_in_4bit)
        
    # Run the model evaluator（调用 `inference` 对象的 `generate_json_completion` 方法，执行函数调用，使用解析后的参数作为输入。）
    inference.generate_json_completion(args.query, args.chat_template, args.max_depth)

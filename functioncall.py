# 添加第三方库argparse：用来处理命令行参数相关的内容
import argparse
# 添加第三方库torch：用来处理LLM模型相关的内容
import torch
# 添加第三方库json：用来处理JSON相关的内容
import json
# 添加第三方库transformers中的AutoModelForCausalLM、AutoTokenizer、BitsAndBytesConfig类
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
# 添加当前项目中的functions
import functions
# 从第三方库prompter中导入PromptManager类
from prompter import PromptManager
# 从第三方库validator中导入validate_function_call_schema类
from validator import validate_function_call_schema
# 从utils中导入多个函数
from utils import (
    print_nous_text_art,
    inference_logger,
    get_assistant_message,
    get_chat_template,
    validate_and_extract_tool_calls
)


# 定义名为ModelInference的类：这个类的作用就是用来model inference
class ModelInference:
    """
    一、ModelInference类的属性成员
        1. `prompter`: 管理对话提示的对象，负责生成对话模板。
        2. `bnb_config`: 配置量化参数，用于控制模型的量化操作。
        3. `model`: 加载的预训练语言生成模型（AutoModelForCausalLM）。
        4. `tokenizer`: 对应模型的分词器，用于文本的分词和编码。
    二、ModelInference类的方法成员
        1. `__init__(self, model_path, chat_template, load_in_4bit)`: 构造函数，初始化模型、分词器和配置。
           - **作用**: 加载和配置模型及其分词器，设置量化参数。
        2. `process_completion_and_validate(self, completion, chat_template)`: 处理模型生成的文本并验证其中的工具调用。
           - **作用**: 解析模型回答，验证工具调用是否符合预期，并处理错误信息。
        3. `execute_function_call(self, tool_call)`: 执行从文本中解析出的工具调用。
           - **作用**: 调用指定的函数，并处理返回结果。
        4. `run_inference(self, prompt)`: 对给定的提示文本执行模型推理。
           - **作用**: 使用模型生成对话回答。
        5. `generate_function_call(self, query, chat_template, num_fewshot, max_depth)`: 根据用户查询生成功能调用并递归处理。
           - **作用**: 管理整个生成和处理对话的流程，包括递归生成和执行功能调用。
    """
    # ModelInference类的构造函数
    def __init__(self, model_path, chat_template, load_in_4bit):
        inference_logger.info(print_nous_text_art())
        self.prompter = PromptManager()
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

    def process_completion_and_validate(self, completion, chat_template):

        assistant_message = get_assistant_message(completion, chat_template, self.tokenizer.eos_token)

        if assistant_message:
            validation, tool_calls, error_message = validate_and_extract_tool_calls(assistant_message)

            if validation:
                inference_logger.info(f"parsed tool calls:\n{json.dumps(tool_calls, indent=2)}")
                return tool_calls, assistant_message, error_message
            else:
                tool_calls = None
                return tool_calls, assistant_message, error_message
        else:
            inference_logger.warning("Assistant message is None")
            raise ValueError("Assistant message is None")
        
    def execute_function_call(self, tool_call):
        function_name = tool_call.get("name")
        function_to_call = getattr(functions, function_name, None)
        function_args = tool_call.get("arguments", {})

        inference_logger.info(f"Invoking function call {function_name} ...")
        function_response = function_to_call(*function_args.values())
        results_dict = f'{{"name": "{function_name}", "content": {function_response}}}'
        return results_dict
    
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

    # 函数接受四个参数：查询字符串 `query`、聊天模板 `chat_template`、少量样本数量 `num_fewshot`，以及一个默认参数 `max_depth` 表示递归调用的最大深度。
    def generate_function_call(self, query, chat_template, num_fewshot, max_depth=5):
        # 开始一个 `try` 语句块，用于捕获和处理任何可能发生的异常
        try:
            # 初始化递归深度为0
            depth = 0
            # 创建初始用户消息，包含查询和一段说明，说明这是第一轮对话，还没有工具的结果可以分析。
            user_message = f"{query}\nThis is the first turn and you don't have <tool_results> to analyze yet"
            # 构建一个包含初始用户消息的聊天记录列表。
            chat = [{"role": "user", "content": user_message}]
            # 调用 `get_openai_tools` 函数，获取可用的 OpenAI 工具。
            tools = functions.get_openai_tools()
            # 调用 `prompter` 对象的 `generate_prompt` 方法，生成聊天提示，这将用于模型推理。
            prompt = self.prompter.generate_prompt(chat, tools, num_fewshot)
            # 执行模型推理，根据生成的提示获得完成文本。
            completion = self.run_inference(prompt)
            # 定义内部递归函数，用于处理递归调用逻辑。
            def recursive_loop(prompt, completion, depth):
                # 声明 `max_depth` 为非局部变量，允许在函数内部修改外部作用域中的 `max_depth`。
                nonlocal max_depth
                # 调用 `process_completion_and_validate` 方法处理模型的输出，并验证
                tool_calls, assistant_message, error_message = self.process_completion_and_validate(completion, chat_template)
                # 将助手的回复添加到对话提示列表中。
                prompt.append({"role": "assistant", "content": assistant_message})
                # 创建一个包含递归深度和用户查询的消息。
                tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
                # 检查是否有工具调用请求。
                if tool_calls:
                    # 接下来的代码块处理每个工具的调用，验证工具的调用是否符合预期的模式，执行工具调用，并根据执行结果更新提示。错误处理和记录也在此处执行。
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")

                    for tool_call in tool_calls:
                        validation, message = validate_function_call_schema(tool_call, tools)
                        if validation:
                            try:
                                function_response = self.execute_function_call(tool_call)
                                tool_message += f"<tool_response>\n{function_response}\n</tool_response>\n"
                                inference_logger.info(f"Here's the response from the function call: {tool_call.get('name')}\n{function_response}")
                            except Exception as e:
                                inference_logger.info(f"Could not execute function: {e}")
                                tool_message += f"<tool_response>\nThere was an error when executing the function: {tool_call.get('name')}\nHere's the error traceback: {e}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                        else:
                            inference_logger.info(message)
                            tool_message += f"<tool_response>\nThere was an error validating function call against function signature: {tool_call.get('name')}\nHere's the error traceback: {message}\nPlease call this function again with correct arguments within XML tags <tool_call></tool_call>\n</tool_response>\n"
                    prompt.append({"role": "tool", "content": tool_message})
                    # 增加递归深度
                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return
                    # 运行模型推理以生成新的完成内容
                    completion = self.run_inference(prompt)
                    # 递归调用 `recursive_loop` 函数继续处理
                    recursive_loop(prompt, completion, depth)
                elif error_message:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")
                    tool_message += f"<tool_response>\nThere was an error parsing function calls\n Here's the error stack trace: {error_message}\nPlease call the function again with correct syntax<tool_response>"
                    prompt.append({"role": "tool", "content": tool_message})

                    depth += 1
                    if depth >= max_depth:
                        print(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
                        return

                    completion = self.run_inference(prompt)
                    recursive_loop(prompt, completion, depth)
                else:
                    inference_logger.info(f"Assistant Message:\n{assistant_message}")

            recursive_loop(prompt, completion, depth)

        # 最后的 `except` 块捕获任何异常，记录错误信息，并抛出异常。
        except Exception as e:
            inference_logger.error(f"Exception occurred: {e}")
            raise e


# 这行代码用于检查当前模块是否为主程序运行环境。只有当文件作为主程序执行时（不是作为模块被导入到其他文件中），代码块内的内容才会执行。
if __name__ == "__main__":
    # 这行代码创建了一个 `ArgumentParser` 对象，该对象用于解析命令行参数。`description` 参数提供了这个程序的简短描述。
    parser = argparse.ArgumentParser(description="Run recursive function calling loop")
    # 向解析器添加一个命令行参数 `--model_path`，这是一个字符串类型的参数，用于指定模型文件的路径。
    parser.add_argument("--model_path", type=str, help="Path to the model folder")
    # 添加一个 `--chat_template` 参数，它也是字符串类型，有一个默认值 `"chatml"`。此参数用于指定聊天提示的格式模板。
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template for prompt formatting")
    # 添加 `--num_fewshot` 参数，这是一个整数类型参数，用来指定在使用 `few-shot` 学习模式时的样本数。默认值为 `None`，表示不使用 `few-shot`。
    parser.add_argument("--num_fewshot", type=int, default=None, help="Option to use json mode examples")
    # 添加 `--load_in_4bit` 参数，它是一个字符串类型的参数，用于决定是否使用 `bitsandbytes` 库将模型加载为 4bit 格式。默认为 `"False"`。
    parser.add_argument("--load_in_4bit", type=str, default="False", help="Option to load in 4bit with bitsandbytes")
    # 添加 `--query` 参数，它是一个字符串类型，具有默认值，用于定义需要处理的默认查询。
    parser.add_argument("--query", type=str, default="I need the current stock price of Tesla (TSLA)")
    # 添加 `--max_depth` 参数，这是一个整数类型的参数，用来设置递归调用的最大深度，默认为 5。
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum number of recursive iteration")
    # 这行代码解析命令行输入的参数，并将解析后的参数存储在 `args` 对象中。
    args = parser.parse_args()

    # specify custom model path（如果 `model_path` 存在，使用提供的参数创建一个 `ModelInference` 对象）
    if args.model_path:
        inference = ModelInference(args.model_path, args.chat_template, args.load_in_4bit)
    else:
        # 设置默认的模型路径。
        model_path = 'NousResearch/Hermes-2-Pro-Llama-3-8B'
        # 使用默认的模型路径创建一个新的 `ModelInference` 对象。
        inference = ModelInference(model_path, args.chat_template, args.load_in_4bit)
        
    # Run the model evaluator（调用 `inference` 对象的 `generate_function_call` 方法，执行函数调用，使用解析后的参数作为输入。）
    inference.generate_function_call(args.query, args.chat_template, args.num_fewshot, args.max_depth)

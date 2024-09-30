import ast
import os
import re
import json
import logging
import datetime
import xml.etree.ElementTree as ET

from art import text2art
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
script_dir = os.path.dirname(os.path.abspath(__file__))
now = datetime.datetime.now()
log_folder = os.path.join(script_dir, "inference_logs")
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(
    log_folder, f"function-calling-inference_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
)
# Use RotatingFileHandler from the logging.handlers module
file_handler = RotatingFileHandler(log_file_path, maxBytes=0, backupCount=0)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S")
file_handler.setFormatter(formatter)

inference_logger = logging.getLogger("function-calling-inference")
inference_logger.addHandler(file_handler)

# 该Python代码定义了一个函数 `print_nous_text_art`，其目的是生成并打印一个特定风格的ASCII艺术文本
def print_nous_text_art(suffix=None):
    font = "nancyj"
    ascii_text = "  nousresearch"
    if suffix:
        """
            在Python中，`ascii_text += f"  x  {suffix}"` 这行代码使用了格式化字符串（f-string），它是一种将变量或表达式的值嵌入字符串的方式。
        这里的代码主要做的是将 `suffix` 变量的值附加到 `ascii_text` 字符串的末尾，并在两者之间加上 "  x  " 作为分隔符。
        """
        ascii_text += f"  x  {suffix}"
    ascii_art = text2art(ascii_text, font=font)
    print(ascii_art)

def get_fewshot_examples(num_fewshot):
    """return a list of few shot examples"""
    example_path = os.path.join(script_dir, 'prompt_assets', 'few_shot.json')
    with open(example_path, 'r') as file:
        examples = json.load(file)  # Use json.load with the file object, not the file path
    if num_fewshot > len(examples):
        raise ValueError(f"Not enough examples (got {num_fewshot}, but there are only {len(examples)} examples).")
    return examples[:num_fewshot]

def get_chat_template(chat_template):
    """read chat template from jinja file"""
    template_path = os.path.join(script_dir, 'chat_templates', f"{chat_template}.j2")

    if not os.path.exists(template_path):
        print
        inference_logger.error(f"Template file not found: {chat_template}")
        return None
    try:
        with open(template_path, 'r') as file:
            template = file.read()
        return template
    except Exception as e:
        print(f"Error loading template: {e}")
        return None

"""
    这个函数的目的是从由语言模型生成的完成文本（`completion`）中提取出助理的回复消息。它使用正则表达式匹配不同的对话模板（例如 `zephyr`, `chatml`, 
`vicuna`），以确保正确地解析出助理的部分，并去除结束符（`eos_token`）。如果对应的模板没有实现，会抛出 `NotImplementedError` 异常。函数处理完成后，
返回清理后的助理消息，或在未找到匹配时返回 `None`。
    函数接受三个参数：`completion`（完成文本），`chat_template`（对话模板），`eos_token`（结束符标记）。

示例: 使用 "chatml" 模板
**输入:**
- `completion`: "<|im_start|> user What is the weather today? <|im_start|> assistant The weather is cloudy."
- `chat_template`: "chatml"
- `eos_token`: "<|endoftext|>"

**处理过程:**
1. 同样先清理掉 `completion` 的空白字符。
2. 因为 `chat_template` 是 "chatml"，用适合的正则表达式匹配从 `<|im_start|> assistant` 开始的内容。
3. 正则表达式匹配到 " The weather is cloudy."。
"""
def get_assistant_message(completion, chat_template, eos_token):
    """define and match pattern to find the assistant message"""
    # strip()函数的作用去除 `completion` 字符串两端的空白字符
    completion = completion.strip()

    if chat_template == "zephyr":
        # 使用正则表达式编译一个模式，匹配从 `<|assistant|>` 标记开始，到字符串末尾的内容，忽略任何嵌套的 `<|assistant|>` 标记
        assistant_pattern = re.compile(r'<\|assistant\|>((?:(?!<\|assistant\|>).)*)$', re.DOTALL)
    elif chat_template == "chatml":
        # 编译一个正则表达式模式，匹配从 `<|im_start|> assistant` 开始到字符串末尾的内容，忽略任何嵌套的 `<|im_start|> assistant`
        assistant_pattern = re.compile(r'<\|im_start\|>\s*assistant((?:(?!<\|im_start\|>\s*assistant).)*)$', re.DOTALL)
    elif chat_template == "vicuna":
        # 编译一个正则表达式模式，匹配从 "ASSISTANT:" 开始到字符串末尾的内容，忽略任何嵌套的 "ASSISTANT:"
        assistant_pattern = re.compile(r'ASSISTANT:\s*((?:(?!ASSISTANT:).)*)$', re.DOTALL)
    else:
        # 抛出 `NotImplementedError` 异常，表明对于未实现的对话模板没有处理方法
        raise NotImplementedError(f"Handling for chat_template '{chat_template}' is not implemented.")
    
    # 在 `completion` 中搜索 `assistant_pattern` 指定的模式
    assistant_match = assistant_pattern.search(completion)
    if assistant_match:
        # 提取匹配的第一个组（即助理消息部分），并去除两端的空白字符
        assistant_content = assistant_match.group(1).strip()
        if chat_template == "vicuna":
            # 将 `eos_token` 格式调整为 `<|vicuna|>` 特定的结束标记
            eos_token = f"</s>{eos_token}"
        return assistant_content.replace(eos_token, "")
    else:
        assistant_content = None
        inference_logger.info("No match found for the assistant pattern")
        return assistant_content

"""
    用于从一段助理内容中提取和验证工具调用的JSON数据。它通过尝试将输入内容视为XML格式进行解析，并在XML结构中查找名为 "tool_call" 的元素。
对于找到的每个元素，它尝试将文本内容解析为JSON对象。如果标准的 `json.loads` 方法失败，则尝试使用 `ast.literal_eval` 作为备选方法。函数
在处理过程中会记录错误，并在发现有效的JSON数据时返回验证结果为真（True）。
"""
def validate_and_extract_tool_calls(assistant_content):
    validation_result = False
    tool_calls = []
    error_message = None

    # 用于捕获在解析和处理数据时可能出现的错误
    try:
        # wrap content in root element（将输入内容封装在一个XML根元素 `<root>` 中，以便能够使用XML解析器进行解析）
        xml_root_element = f"<root>{assistant_content}</root>"
        # 使用 `ET.fromstring` 方法解析封装好的XML字符串，创建一个XML树的根节点
        root = ET.fromstring(xml_root_element)

        # extract JSON data（遍历根节点下所有名为 "tool_call" 的元素）
        for element in root.findall(".//tool_call"):
            json_data = None
            # 开始内部的异常处理块，用于处理文本到JSON的转换
            try:
                # 提取元素的文本内容，并去除首尾空格
                json_text = element.text.strip()

                # 尝试使用 `json.loads` 将文本转换为JSON对象
                try:
                    # Prioritize json.loads for better error handling（如果文本是有效的JSON格式，则将解析结果赋值给 `json_data`）
                    json_data = json.loads(json_text)
                except json.JSONDecodeError as json_err:
                    # 在 `json.loads` 失败后，尝试使用 `ast.literal_eval` 作为备选方法
                    try:
                        # Fallback to ast.literal_eval if json.loads fails
                        json_data = ast.literal_eval(json_text)
                    except (SyntaxError, ValueError) as eval_err:
                        error_message = f"JSON parsing failed with both json.loads and ast.literal_eval:\n"\
                                        f"- JSON Decode Error: {json_err}\n"\
                                        f"- Fallback Syntax/Value Error: {eval_err}\n"\
                                        f"- Problematic JSON text: {json_text}"
                        inference_logger.error(error_message)
                        continue
            except Exception as e:
                error_message = f"Cannot strip text: {e}"
                inference_logger.error(error_message)
            # 如果成功解析出JSON数据，则更新 `tool_calls` 列表，并将验证结果设为真（True）
            if json_data is not None:
                tool_calls.append(json_data)
                validation_result = True
    # 捕获并处理XML解析错误
    except ET.ParseError as err:
        error_message = f"XML Parse Error: {err}"
        inference_logger.error(f"XML Parse Error: {err}")

    # Return default values if no valid data is extracted
    return validation_result, tool_calls, error_message

def extract_json_from_markdown(text):
    """
    Extracts the JSON string from the given text using a regular expression pattern.
    
    Args:
        text (str): The input text containing the JSON string.
        
    Returns:
        dict: The JSON data loaded from the extracted string, or None if the JSON string is not found.
    """
    json_pattern = r'```json\r?\n(.*?)\r?\n```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON string: {e}")
    else:
        print("JSON string not found in the text.")
    return None


import sys
import inspect

def code_interpreter(code_markdown: str) -> dict | str:
    # 尝试执行以下代码块
    try:
        # 从 Markdown 代码块中提取代码
        # 假定代码被包围在 ```python 和 ``` 之间，分别位于第一行和最后一行
        code_lines = code_markdown.split('\n')[1:-1]
        # 将提取的代码行重新组合成一个完整的代码字符串
        code_without_markdown = '\n'.join(code_lines)
        
        # 创建一个新的字典，用作执行代码的命名空间
        exec_namespace = {}

        # 在新的命名空间中执行提取的代码字符串
        # 使用 exec() 函数将字符串作为代码执行，而不返回任何结果
        exec(code_without_markdown, exec_namespace)

        # 初始化一个空字典，用于收集执行结果
        result_dict = {}
        # 遍历执行后的命名空间中的所有变量和函数
        for name, value in exec_namespace.items():
            # 判断该项是否为可调用的函数或方法
            if callable(value):
                try:
                    # 尝试直接调用函数，不带参数
                    result_dict[name] = value()
                except TypeError:
                    # 如果函数调用失败（通常是因为需要参数），尝试传递命名空间中的变量作为参数
                    arg_names = inspect.getfullargspec(value).args
                    args = {arg_name: exec_namespace.get(arg_name) for arg_name in arg_names}
                    result_dict[name] = value(**args)
            elif not name.startswith('_'):  # 排除以 '_' 开头的变量，通常这些是内部变量
                # 将非函数的变量和它们的值添加到结果字典中
                result_dict[name] = value

        # 返回包含所有有效结果的字典
        return result_dict

    except Exception as e:
        # 如果在执行过程中发生任何异常，捕获异常并返回错误消息
        error_message = f"An error occurred: {e}"
        return error_message
    
def main(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            code_markdown = file.read()
        # 使用code_interpreter函数来对code_markdown中的代码进行解析并且执行其中的代码
        result = code_interpreter(code_markdown)
        print(result)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_markdown_file>")
    else:
        main(sys.argv[1])

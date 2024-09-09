# `re` 模块提供正则表达式支持，用于字符串的搜索、替换和解析。正则表达式是用于匹配字符串中字符组合的模式。
import re
# `inspect` 模块用于检查（或检视）活跃的对象，如函数、类、方法、模块等，可以获取这些对象的信息，如源码、变量名、注释等。
import inspect
# `requests` 是一个用于发送 HTTP 请求的库。它是 Python 最常用的网络请求库之一，使发送网络请求变得简洁明了。
import requests
# `pandas` 是一个强大的数据分析和操作库，提供高性能、易用的数据结构和数据分析工具，尤其适合处理表格数据（如 CSV 文件）。
import pandas as pd
# `yfinance` 库提供了一个易于使用的方法来获取雅虎财经的市场数据。用户可以下载历史市场数据、实时数据等。
import yfinance as yf
# `concurrent.futures` 模块提供了一个高级接口，用于异步执行可调用对象。这个模块使得线程或进程的使用和管理变得更加简单。
import concurrent.futures

# `typing` 模块支持 Python 类型标注，用于在代码中添加变量类型提示。`List` 是一个类型类，用于指示变量应为列表类型。
from typing import List
# `BeautifulSoup` 是一个用于解析 HTML 和 XML 文档的库，常用于网页抓取。它创建一个解析树以便提取 XML 和 HTML 标签的数据。
from bs4 import BeautifulSoup
# 这可能是一个自定义的日志记录器，用于记录推理或其他运行时信息。具体功能取决于 `utils` 模块的实现细节。
from utils import inference_logger
# 这些模块和函数看起来是 `langchain` 库的一部分，可能与自然语言处理或链式工具调用有关。`tool` 可能是用于定义工具的装饰器或类，而 `convert_to_openai_tool` 可能是将某些功能转换为兼容 OpenAI 格式的工具的函数。
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

@tool
def code_interpreter(code_markdown: str) -> dict | str:
    """
    Execute the provided Python code string on the terminal using exec.

    The string should contain valid, executable and pure Python code in markdown syntax.
    Code should also import any required Python packages.

    Args:
        code_markdown (str): The Python code with markdown syntax to be executed.
            For example: ```python\n<code-string>\n```

    Returns:
        dict | str: A dictionary containing variables declared and values returned by function calls,
            or an error message if an exception occurred.

    Note:
        Use this function with caution, as executing arbitrary code can pose security risks.
    """
    try:
        # Extracting code from Markdown code block
        code_lines = code_markdown.split('\n')[1:-1]
        code_without_markdown = '\n'.join(code_lines)

        # Create a new namespace for code execution
        exec_namespace = {}

        # Execute the code in the new namespace
        exec(code_without_markdown, exec_namespace)

        # Collect variables and function call results
        result_dict = {}
        for name, value in exec_namespace.items():
            if callable(value):
                try:
                    result_dict[name] = value()
                except TypeError:
                    # If the function requires arguments, attempt to call it with arguments from the namespace
                    arg_names = inspect.getfullargspec(value).args
                    args = {arg_name: exec_namespace.get(arg_name) for arg_name in arg_names}
                    result_dict[name] = value(**args)
            elif not name.startswith('_'):  # Exclude variables starting with '_'
                result_dict[name] = value

        return result_dict

    except Exception as e:
        error_message = f"An error occurred: {e}"
        inference_logger.error(error_message)
        return error_message

@tool
def google_search_and_scrape(query: str) -> dict:
    """
    Performs a Google search for the given query, retrieves the top search result URLs,
    and scrapes the text content and table data from those pages in parallel.

    Args:
        query (str): The search query.
    Returns:
        list: A list of dictionaries containing the URL, text content, and table data for each scraped page.
    """
    num_results = 2
    url = 'https://www.google.com/search'
    params = {'q': query, 'num': num_results}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.3'}
    
    inference_logger.info(f"Performing google search with query: {query}\nplease wait...")
    response = requests.get(url, params=params, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    urls = [result.find('a')['href'] for result in soup.find_all('div', class_='tF2Cxc')]
    
    inference_logger.info(f"Scraping text from urls, please wait...") 
    [inference_logger.info(url) for url in urls]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(lambda url: (url, requests.get(url, headers=headers).text if isinstance(url, str) else None), url) for url in urls[:num_results] if isinstance(url, str)]
        results = []
        for future in concurrent.futures.as_completed(futures):
            url, html = future.result()
            soup = BeautifulSoup(html, 'html.parser')
            paragraphs = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
            text_content = ' '.join(paragraphs)
            text_content = re.sub(r'\s+', ' ', text_content)
            table_data = [[cell.get_text(strip=True) for cell in row.find_all('td')] for table in soup.find_all('table') for row in table.find_all('tr')]
            if text_content or table_data:
                results.append({'url': url, 'content': text_content, 'tables': table_data})
    return results

@tool
def get_current_stock_price(symbol: str) -> float:
  """
  Get the current stock price for a given symbol.

  Args:
    symbol (str): The stock symbol.

  Returns:
    float: The current stock price, or None if an error occurs.
  """
  try:
    stock = yf.Ticker(symbol)
    # Use "regularMarketPrice" for regular market hours, or "currentPrice" for pre/post market
    current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
    return current_price if current_price else None
  except Exception as e:
    print(f"Error fetching current price for {symbol}: {e}")
    return None

@tool
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get fundamental data for a given stock symbol using yfinance API.

    Args:
        symbol (str): The stock symbol.

    Returns:
        dict: A dictionary containing fundamental data.
            Keys:
                - 'symbol': The stock symbol.
                - 'company_name': The long name of the company.
                - 'sector': The sector to which the company belongs.
                - 'industry': The industry to which the company belongs.
                - 'market_cap': The market capitalization of the company.
                - 'pe_ratio': The forward price-to-earnings ratio.
                - 'pb_ratio': The price-to-book ratio.
                - 'dividend_yield': The dividend yield.
                - 'eps': The trailing earnings per share.
                - 'beta': The beta value of the stock.
                - '52_week_high': The 52-week high price of the stock.
                - '52_week_low': The 52-week low price of the stock.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        fundamentals = {
            'symbol': symbol,
            'company_name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('forwardPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'dividend_yield': info.get('dividendYield', None),
            'eps': info.get('trailingEps', None),
            'beta': info.get('beta', None),
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None)
        }
        return fundamentals
    except Exception as e:
        print(f"Error getting fundamentals for {symbol}: {e}")
        return {}

@tool
def get_financial_statements(symbol: str) -> dict:
    """
    Get financial statements for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing financial statements (income statement, balance sheet, cash flow statement).
    """
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        return financials
    except Exception as e:
        print(f"Error fetching financial statements for {symbol}: {e}")
        return {}

@tool
def get_key_financial_ratios(symbol: str) -> dict:
    """
    Get key financial ratios for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing key financial ratios.
    """
    try:
        stock = yf.Ticker(symbol)
        key_ratios = stock.info
        return key_ratios
    except Exception as e:
        print(f"Error fetching key financial ratios for {symbol}: {e}")
        return {}

@tool
def get_analyst_recommendations(symbol: str) -> pd.DataFrame:
    """
    Get analyst recommendations for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing analyst recommendations.
    """
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        return recommendations
    except Exception as e:
        print(f"Error fetching analyst recommendations for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_dividend_data(symbol: str) -> pd.DataFrame:
    """
    Get dividend data for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing dividend data.
    """
    try:
        stock = yf.Ticker(symbol)
        dividends = stock.dividends
        return dividends
    except Exception as e:
        print(f"Error fetching dividend data for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_company_news(symbol: str) -> pd.DataFrame:
    """
    Get company news and press releases for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing company news and press releases.
    """
    try:
        news = yf.Ticker(symbol).news
        return news
    except Exception as e:
        print(f"Error fetching company news for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_technical_indicators(symbol: str) -> pd.DataFrame:
    """
    Get technical indicators for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    pd.DataFrame: DataFrame containing technical indicators.
    """
    try:
        indicators = yf.Ticker(symbol).history(period="max")
        return indicators
    except Exception as e:
        print(f"Error fetching technical indicators for {symbol}: {e}")
        return pd.DataFrame()

@tool
def get_company_profile(symbol: str) -> dict:
    """
    Get company profile and overview for a given stock symbol.

    Args:
    symbol (str): The stock symbol.

    Returns:
    dict: Dictionary containing company profile and overview.
    """
    try:
        profile = yf.Ticker(symbol).info
        return profile
    except Exception as e:
        print(f"Error fetching company profile for {symbol}: {e}")
        return {}

def get_openai_tools() -> List[dict]:
    functions = [
        code_interpreter,
        google_search_and_scrape,
        get_current_stock_price,
        get_company_news,
        get_company_profile,
        get_stock_fundamentals,
        get_financial_statements,
        get_key_financial_ratios,
        get_analyst_recommendations,
        get_dividend_data,
        get_technical_indicators
    ]

    tools = [convert_to_openai_tool(f) for f in functions]
    return tools
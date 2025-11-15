# -*- coding: utf-8 -*-
import sys
import locale

# Set encoding to handle Unicode characters properly
if sys.platform.startswith('win'):
    # Windows specific encoding setup
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
else:
    # Unix/Linux/Mac encoding setup
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass  # Use system default

from flask import Flask, render_template, request, jsonify, send_file, session
import os
import pandas as pd
from pathlib import Path
import json
import re
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List
import base64
import io
from PIL import Image
import akshare as ak
import finnhub
import numpy as np
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def safe_str(obj):
    """Safely convert object to string, handling encoding issues"""
    try:
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, str):
            return obj.encode('utf-8', errors='replace').decode('utf-8')
        else:
            return str(obj).encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return "Error converting to string"


# Import your existing modules
from trading_graph import TradingGraph

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# 添加结果存储字典（在生产环境中应该使用Redis等持久化存储）
analysis_results_store = {}


class MultiProviderLLM:
    """支持多厂商LLM API的类"""

    def __init__(self):
        self.providers = {
            'openai': {
                'name': 'OpenAI',
                'client_class': OpenAIClient,
                'base_url': None,
                'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
            },
            'deepseek': {
                'name': 'DeepSeek',
                'client_class': OpenAIClient,
                'base_url': 'https://api.deepseek.com/v1',
                'models': ['deepseek-chat', 'deepseek-coder']
            },
            'volcengine': {
                'name': 'Volcengine',
                'client_class': OpenAIClient,
                'base_url': 'https://ark-cn-beijing.bytedance.net/api/v3',
                'models': ['ep-20250519162223-96wj4']
            }
        }
        self.current_provider = 'volcengine'
        self.api_key = os.environ.get("VOLCENGINE_API_KEY", "")

    def set_provider(self, provider: str, api_key: str = None):
        """Set current LLM provider"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")

        self.current_provider = provider
        if api_key:
            self.api_key = api_key

        # Set environment variables
        if provider == 'openai':
            os.environ["OPENAI_API_KEY"] = self.api_key
        elif provider == 'deepseek':
            os.environ["DEEPSEEK_API_KEY"] = self.api_key
        elif provider == 'volcengine':
            os.environ["VOLCENGINE_API_KEY"] = self.api_key

    def get_client(self):
        """获取当前配置的LLM客户端"""
        provider_config = self.providers[self.current_provider]
        client_class = provider_config['client_class']

        kwargs = {'api_key': self.api_key}
        if provider_config['base_url']:
            kwargs['base_url'] = provider_config['base_url']

        return client_class(**kwargs)

    def validate_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """Validate API key"""
        try:
            if provider not in self.providers:
                return {"valid": False, "error": f"Unsupported provider: {provider}"}

            # Use first model from provider config for validation
            provider_config = self.providers[provider]
            model = provider_config['models'][0]  # 使用第一个可用模型

            if provider == 'openai':
                client = OpenAIClient(api_key=api_key)
            elif provider == 'deepseek':
                client = OpenAIClient(api_key=api_key, base_url=provider_config['base_url'])
            elif provider == 'volcengine':
                client = OpenAIClient(api_key=api_key, base_url=provider_config['base_url'])
            else:
                return {"valid": False, "error": f"Unsupported provider: {provider}"}

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return {"valid": True, "message": f"{provider_config['name']} API key is valid"}

        except Exception as e:
            error_msg = safe_str(e)

            if "authentication" in error_msg.lower() or "invalid api key" in error_msg.lower() or "401" in error_msg:
                return {"valid": False, "error": f"Invalid {self.providers[provider]['name']} API key"}
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {"valid": False, "error": "API rate limit exceeded, please try again later"}
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {"valid": False, "error": "Account quota exceeded or billing issue"}
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {"valid": False, "error": "Network connection error"}
            elif "model not exist" in error_msg.lower() or "400" in error_msg:
                return {"valid": False, "error": f"Model does not exist: {error_msg}"}
            else:
                return {"valid": False, "error": f"API validation error: {error_msg}"}


class MultiSourceDataFetcher:
    """支持多数据源的类"""

    def __init__(self):
        self.sources = ['akshare', 'finnhub', 'yfinance']
        self.current_source = 'akshare'
        self.finnhub_client = None

    def initialize_finnhub(self, api_key: str = None):
        """Initialize Finnhub client"""
        api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if api_key and api_key != "your-finnhub-api-key-here":
            self.finnhub_client = finnhub.Client(api_key=api_key)

    def fetch_akshare_data(self, symbol: str, period: str = "daily",
                           start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch stock data using akshare"""
        try:
            print(f"正在获取 {symbol} 的数据...")
            print(f"  开始日期: {start_date}, 结束日期: {end_date}")

            # 先尝试获取真实数据，如果失败再使用demo数据
            df = pd.DataFrame()

            # akshare数据获取逻辑 - 尝试多种方法
            # 根据股票代码类型选择不同的函数
            functions_to_try = []

            # 美股代码（通常为1-5个字母）
            if re.match(r'^[A-Z]{1,5}$', symbol):
                print(f"  检测到美股代码: {symbol}")
                functions_to_try.extend([
                    ('stock_us_daily', symbol),  # 美股日线数据
                    ('stock_us_spot', symbol),  # 美股实时数据
                ])
            # A股代码（6位数字）
            elif re.match(r'^\d{6}$', symbol):
                print(f"  检测到A股代码: {symbol}")
                functions_to_try.extend([
                    ('stock_zh_a_hist', symbol),
                    ('stock_zh_index_daily_em', symbol),
                ])
            # 指数代码
            elif symbol.startswith('SH') or symbol.startswith('SZ'):
                print(f"  检测到指数代码: {symbol}")
                functions_to_try.extend([
                    ('index_zh_a_hist', symbol),
                    ('stock_zh_index_daily_em', symbol),
                ])
            else:
                print(f"  未知代码格式: {symbol}，尝试所有方法")
                functions_to_try.extend([
                    ('stock_zh_index_daily_em', symbol),
                    ('stock_zh_a_hist', symbol),
                    ('index_zh_a_hist', symbol),
                    ('stock_us_daily', symbol),
                    ('stock_us_spot', symbol),
                ])

            for func_name, sym in functions_to_try:
                try:
                    func = getattr(ak, func_name)
                    if func_name == 'stock_zh_a_hist':
                        df = func(symbol=sym, period="daily",
                                  start_date=start_date.replace('-', ''),
                                  end_date=end_date.replace('-', ''), adjust="")
                    elif func_name == 'index_zh_a_hist':
                        df = func(symbol=sym, period="daily",
                                  start_date=start_date.replace('-', ''),
                                  end_date=end_date.replace('-', ''))
                    elif func_name == 'stock_us_daily':
                        # 美股日线数据
                        df = func(symbol=sym)
                    elif func_name == 'stock_us_spot':
                        # 美股实时数据，可能需要转换为日线
                        df = func()
                        if not df.empty and symbol in df['symbol'].values:
                            df = df[df['symbol'] == symbol]
                    else:
                        df = func(symbol=sym)

                    # 检查df是否为None或空
                    if df is None:
                        print(f"{func_name} 返回None，跳过")
                        continue

                    if not df.empty:
                        print(f"使用 {func_name} 成功获取数据，数据形状: {df.shape}")
                        break
                    else:
                        print(f"{func_name} 返回空DataFrame")

                except Exception as e:
                    error_msg = safe_str(e)
                    print(f"{func_name} 失败: {error_msg}")
                    if "subscriptable" in error_msg:
                        print(f"  详细错误: 函数 {func_name} 可能返回了None")
                    continue

            if df.empty:
                print(f"所有akshare方法都失败，使用demo数据")
                return self.get_demo_data(symbol, start_date, end_date)

            # Standardize column names - 更全面的映射
            column_mapping = {
                'date': 'Datetime', '日期': 'Datetime', 'Date': 'Datetime',
                'open': 'Open', '开盘': 'Open', 'Open': 'Open',
                'high': 'High', '最高': 'High', 'High': 'High',
                'low': 'Low', '最低': 'Low', 'Low': 'Low',
                'close': 'Close', '收盘': 'Close', 'Close': 'Close',
                'volume': 'Volume', '成交量': 'Volume', 'Volume': 'Volume',
                '成交额': 'Volume', 'amount': 'Volume'
            }

            # 应用列名映射
            print(f"  原始列名: {list(df.columns)}")
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            print(f"  映射后列名: {list(df.columns)}")

            # Ensure Datetime column exists
            if 'Datetime' not in df.columns:
                if df.index.name in ['date', '日期', 'Date']:
                    df = df.reset_index()
                    df = df.rename(columns={df.columns[0]: 'Datetime'})
                elif len(df.columns) > 0 and any(col.lower() in ['date', 'datetime', '日期'] for col in df.columns):
                    # 找到日期列
                    date_col = next((col for col in df.columns if col.lower() in ['date', 'datetime', '日期']), None)
                    if date_col:
                        df = df.rename(columns={date_col: 'Datetime'})

            # Convert Datetime column to datetime
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime')


            # Filter by date range
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                print(f"start_dt and end_dt: {start_dt},{end_dt}，使用demo数据")

            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"缺少必要的列: {missing_columns}，使用demo数据")
                return self.get_demo_data(symbol, start_date, end_date)

            print(f"成功获取 {len(df)} 条数据")
            print(f"成功获取 {df} ")
            return df

        except Exception as e:
            error_msg = safe_str(e)
            print(f"akshare数据获取失败: {error_msg}，使用demo数据")
            return self.get_demo_data(symbol, start_date, end_date)

    def get_demo_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成demo数据用于测试"""
        try:
            # 生成日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            dates = pd.date_range(start=start_dt, end=end_dt, freq='D')

            # 过滤工作日
            dates = dates[dates.weekday < 5]  # 0-4 是周一到周五

            if len(dates) == 0:
                return pd.DataFrame()

            # 生成模拟价格数据，使用股票名称的哈希值作为种子，确保不同股票生成不同数据
            seed = hash(symbol) % 10000
            np.random.seed(seed)

            # 基础价格 - 根据股票名称生成更有差异化的基础价格
            if symbol.upper() in ['BTC', 'ETH', 'SOL']:
                # 加密货币价格较高
                base_price = 50000 + (hash(symbol) % 10000)
            elif symbol.upper() in ['AAPL', 'MSFT', 'GOOGL']:
                # 大型科技股
                base_price = 150 + (hash(symbol) % 100)
            elif re.match(r'^\d{6}$', symbol):
                # 可能是A股代码
                base_price = 10 + (hash(symbol) % 90)
            else:
                # 其他股票
                base_price = 50 + (hash(symbol) % 150)

            # 生成价格序列
            returns = np.random.normal(0, 0.02, len(dates))  # 2%日波动率
            prices = [base_price]

            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # 生成OHLC数据
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                daily_volatility = 0.01  # 1%日内波动
                high = price * (1 + np.random.uniform(0, daily_volatility))
                low = price * (1 - np.random.uniform(0, daily_volatility))
                open_price = prices[i - 1] if i > 0 else price
                close_price = price
                volume = np.random.randint(1000000, 10000000)

                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close_price,
                    'Volume': volume
                })

            df = pd.DataFrame(data, index=dates)
            df.index.name = 'Datetime'

            # 重置索引，将Datetime作为列
            df = df.reset_index()

            print(f"生成了 {len(df)} 条 {symbol} 的demo数据")
            return df

        except Exception as e:
            print(f"生成demo数据失败: {safe_str(e)}")
            return pd.DataFrame()

    def fetch_finnhub_data(self, symbol: str, resolution: str = 'D',
                           start_timestamp: int = None, end_timestamp: int = None) -> pd.DataFrame:
        """Fetch stock data using Finnhub"""
        try:
            if not self.finnhub_client:
                self.initialize_finnhub()
                if not self.finnhub_client:
                    return pd.DataFrame()

            # Use default range if no timestamp provided
            if not start_timestamp:
                start_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())
            if not end_timestamp:
                end_timestamp = int(datetime.now().timestamp())

            # Get candlestick data
            data = self.finnhub_client.stock_candles(symbol, resolution,
                                                     start_timestamp, end_timestamp)

            if not data or data.get('s') != 'ok':
                return pd.DataFrame()

            # 转换为DataFrame
            df = pd.DataFrame({
                'Datetime': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            })

            return df

        except Exception as e:
            print(f"Finnhub data fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_data(self, symbol: str, timeframe: str,
                   start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from multiple sources with priority fallback"""
        df = pd.DataFrame()

        # 首先尝试akshare
        if self.current_source == 'akshare' or df.empty:
            df = self.fetch_akshare_data(symbol, self._convert_timeframe(timeframe),
                                         start_date.strftime('%Y%m%d'),
                                         end_date.strftime('%Y%m%d'))

        # 如果akshare失败，尝试finnhub
        if df.empty and self.finnhub_client:
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            df = self.fetch_finnhub_data(symbol, self._convert_timeframe(timeframe),
                                         start_ts, end_ts)

        return df

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe format"""
        timeframe_map = {
            '1m': '1', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '4h': '240', '1d': 'D', '1w': 'W', '1M': 'M'
        }
        return timeframe_map.get(timeframe, 'D')


class WebTradingAnalyzer:
    def __init__(self):
        """Initialize the web trading analyzer with multi-provider support."""
        self.data_dir = Path("data")
        self.llm_provider = MultiProviderLLM()
        self.data_fetcher = MultiSourceDataFetcher()

        # 根据环境变量设置LLM提供商
        llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")
        if llm_provider == "deepseek":
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
            if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
                self.llm_provider.set_provider('deepseek', deepseek_key)
        elif llm_provider == "openai":
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key and openai_key != "your-openai-api-key-here":
                self.llm_provider.set_provider('openai', openai_key)
        elif llm_provider == "volcengine":
            volcengine_key = os.environ.get("VOLCENGINE_API_KEY")
            if volcengine_key and volcengine_key != "your-volcengine-api-key-here":
                self.llm_provider.set_provider('volcengine', volcengine_key)

        # 初始化TradingGraph
        self.trading_graph = TradingGraph()

        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Available assets and their display names
        self.asset_mapping = {
            'SPX': 'S&P 500',
            'BTC': 'Bitcoin',
            'GC': 'Gold Futures',
            'NQ': 'Nasdaq Futures',
            'CL': 'Crude Oil',
            'ES': 'E-mini S&P 500',
            'DJI': 'Dow Jones',
            'QQQ': 'Invesco QQQ Trust',
            'VIX': 'Volatility Index',
            'DXY': 'US Dollar Index',
            'AAPL': 'Apple Inc.',
            'TSLA': 'Tesla Inc.',
            '000001': 'Shanghai Composite Index',
            '399001': 'Shenzhen Component Index',
            'SH000300': 'CSI 300 Index',
            'SH510300': 'CSI 300 ETF',
        }

        # Symbol mapping for different data sources
        self.symbol_mapping = {
            'akshare': {
                'SPX': 'SPX',
                'BTC': 'BTC-USD',
                'AAPL': 'AAPL',
                'TSLA': 'TSLA',
                '000001': '000001',
                '399001': '399001',
                'SH000300': '000300',
                'SH510300': '510300',
            },
            'finnhub': {
                'SPX': 'SPX',
                'BTC': 'BINANCE:BTCUSDT',
                'AAPL': 'AAPL',
                'TSLA': 'TSLA',
            },
            'yfinance': {
                'SPX': '^GSPC',
                'BTC': 'BTC-USD',
                'GC': 'GC=F',
                'NQ': 'NQ=F',
                'CL': 'CL=F',
                'ES': 'ES=F',
                'DJI': '^DJI',
                'QQQ': 'QQQ',
                'VIX': '^VIX',
                'DXY': 'DX-Y.NYB',
                'AAPL': 'AAPL',
                'TSLA': 'TSLA',
            }
        }

        # Timeframe mapping
        self.timeframe_mapping = {
            'akshare': {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '4h': '240', '1d': 'daily', '1w': 'weekly', '1M': 'monthly'
            },
            'finnhub': {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '4h': '240', '1d': 'D', '1w': 'W', '1M': 'M'
            },
            'yfinance': {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1M': '1mo'
            }
        }

        # Available timeframes
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()

        # Initialize Finnhub with environment variable
        self.data_fetcher.initialize_finnhub()

    def fetch_market_data(self, symbol: str, interval: str,
                          start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from multiple sources with fallback."""
        # Try akshare first
        df = self.data_fetcher.fetch_akshare_data(
            self.symbol_mapping['akshare'].get(symbol, symbol),
            self.timeframe_mapping['akshare'].get(interval, 'daily'),
            start_datetime.strftime('%Y%m%d'),
            end_datetime.strftime('%Y%m%d')
        )

        # If akshare fails, try finnhub
        if df.empty:
            start_ts = int(start_datetime.timestamp())
            end_ts = int(end_datetime.timestamp())
            df = self.data_fetcher.fetch_finnhub_data(
                self.symbol_mapping['finnhub'].get(symbol, symbol),
                self.timeframe_mapping['finnhub'].get(interval, 'D'),
                start_ts, end_ts
            )

        # If both fail, provide empty DataFrame
        if df.empty:
            print(f"All data sources failed to fetch data for {symbol}")

        return df

    # Keep other methods unchanged, only modify data fetching part
    def run_analysis(self, df: pd.DataFrame, asset_name: str, timeframe: str, generate_charts: bool = False,
                     trading_strategy: str = 'high_frequency') -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            # 确保asset_name是安全的字符串
            asset_name = safe_str(asset_name)
            timeframe = safe_str(timeframe)

            # if len(df) > 49:
            #     df_slice = df.tail(49).iloc[:-3]
            # else:
            #     df_slice = df.tail(45)

            # 保留最后49条数据
            if len(df) > 49:
                df_slice = df.tail(49)  # 只取最后49条，不去掉最后3条
            else:
                df_slice = df.copy()

            # 检查必要的列，注意Datetime可能被设置为索引
            required_price_columns = ["Open", "High", "Low", "Close"]

            # 检查Datetime列是否存在，或者是否在索引中
            has_datetime_column = "Datetime" in df_slice.columns
            has_datetime_index = df_slice.index.name == "Datetime" or isinstance(df_slice.index, pd.DatetimeIndex)

            if not all(col in df_slice.columns for col in required_price_columns) or (
                    not has_datetime_column and not has_datetime_index):
                return {
                    "success": False,
                    "error": f"Missing required columns. Available columns: {list(df_slice.columns)}, Index: {df_slice.index.name}"
                }

            # 处理Datetime数据（可能在列中或索引中）
            df_slice_dict = {}

            # 如果Datetime在索引中，重置索引
            if has_datetime_index:
                df_slice = df_slice.reset_index()
                has_datetime_column = True

            # 处理Datetime列
            if has_datetime_column:
                try:
                    df_slice_dict['Datetime'] = df_slice['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                except Exception:
                    df_slice_dict['Datetime'] = [safe_str(dt) for dt in df_slice['Datetime'].tolist()]

            # 处理价格数据列
            for col in required_price_columns:
                try:
                    df_slice_dict[col] = [float(x) if pd.notna(x) else 0.0 for x in df_slice[col].tolist()]
                except Exception:
                    df_slice_dict[col] = [safe_str(x) for x in df_slice[col].tolist()]

            display_timeframe = timeframe
            if timeframe.endswith('h'):
                display_timeframe += 'our'
            elif timeframe.endswith('m'):
                display_timeframe += 'in'
            elif timeframe.endswith('d'):
                display_timeframe += 'ay'

            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": safe_str(display_timeframe),
                "stock_name": safe_str(asset_name),
                "trading_strategy": safe_str(trading_strategy)  # Add trading strategy to state
            }

            # 根据generate_charts参数决定是否生成图表
            if generate_charts:
                analysis_result = self.trading_graph.analyze(df_slice_dict, asset_name, display_timeframe,
                                                             trading_strategy)
            else:
                # 只进行文本分析，不生成图表
                analysis_result = self.trading_graph.analyze_text_only(df_slice_dict, asset_name, display_timeframe,
                                                                       trading_strategy)

            # 从分析结果中提取final_state
            final_state = analysis_result.get("final_state", {})
            print(f"从TradingGraph提取的final_state键: {list(final_state.keys())}")
            print(f"趋势报告长度: {len(final_state.get('trend_report', ''))}")
            print(f"指标报告长度: {len(final_state.get('indicator_report', ''))}")
            print(f"形态报告长度: {len(final_state.get('pattern_report', ''))}")

            # 确保final_state中的所有字符串都是安全的
            if isinstance(final_state, dict):
                for key, value in final_state.items():
                    if isinstance(value, str):
                        final_state[key] = safe_str(value)

            return {
                "success": True,
                "final_state": final_state,
                "asset_name": safe_str(asset_name),
                "timeframe": safe_str(display_timeframe),
                "data_length": len(df_slice)
            }

        except Exception as e:
            error_msg = safe_str(e)

            if "authentication" in error_msg.lower():
                return {"success": False, "error": "API key invalid"}
            elif "rate limit" in error_msg.lower():
                return {"success": False, "error": "API rate limit exceeded"}
            else:
                return {"success": False, "error": f"Analysis error: {error_msg}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current API key for the specified provider."""
        if provider:
            return self.llm_provider.validate_api_key(provider, os.environ.get(f"{provider.upper()}_API_KEY", ""))

        # Default to validate current provider
        current_key = self.llm_provider.api_key
        return self.llm_provider.validate_api_key(self.llm_provider.current_provider, current_key)

    # Keep other helper methods unchanged
    def get_available_assets(self) -> list:
        return sorted(list(self.asset_mapping.keys()))

    def load_custom_assets(self) -> list:
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 确保所有资产名称都是安全字符串
                    return [safe_str(asset) for asset in data if asset]
            return []
        except Exception as e:
            print(f"Failed to load custom assets: {safe_str(e)}")
            return []

    def save_custom_asset(self, symbol: str) -> bool:
        try:
            symbol = safe_str(symbol).strip()
            if not symbol or symbol in self.custom_assets:
                return True
            self.custom_assets.append(symbol)
            with open(self.custom_assets_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_assets, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save custom asset: {safe_str(e)}")
            return False

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {"error": safe_str(results["error"])}

        final_state = results["final_state"]

        # Extract analysis results from state fields with safe string conversion
        technical_indicators = safe_str(final_state.get("indicator_report", ""))
        pattern_analysis = safe_str(final_state.get("pattern_report", ""))
        trend_analysis = safe_str(final_state.get("trend_report", ""))
        final_decision_raw = safe_str(final_state.get("final_trade_decision", ""))

        # Extract chart data if available
        pattern_chart = safe_str(final_state.get("pattern_image", ""))
        trend_chart = safe_str(final_state.get("trend_image", ""))
        pattern_image_filename = safe_str(final_state.get("pattern_image_filename", ""))
        trend_image_filename = safe_str(final_state.get("trend_image_filename", ""))

        # Parse final decision
        final_decision = ""
        if final_decision_raw:
            try:
                # Try to extract JSON from the decision
                start = final_decision_raw.find('{')
                end = final_decision_raw.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = final_decision_raw[start:end]
                    decision_data = json.loads(json_str)
                    final_decision = {
                        "decision": safe_str(decision_data.get('decision', 'N/A')),
                        "risk_reward_ratio": safe_str(decision_data.get('risk_reward_ratio', 'N/A')),
                        "forecast_horizon": safe_str(decision_data.get('forecast_horizon', 'N/A')),
                        "justification": safe_str(decision_data.get('justification', 'N/A'))
                    }
                else:
                    # If no JSON found, return the raw text
                    final_decision = {"raw": safe_str(final_decision_raw)}
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                final_decision = {"raw": safe_str(final_decision_raw)}

        return {
            "success": True,
            "asset_name": safe_str(results["asset_name"]),
            "timeframe": safe_str(results["timeframe"]),
            "data_length": results["data_length"],
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision
        }


# Initialize the analyzer
analyzer = WebTradingAnalyzer()


# Setup environment variables (if they exist)
def setup_environment():
    """Setup environment variables"""
    # Read API keys from environment variables or config file
    llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")

    # Set LLM provider based on environment variables
    if llm_provider == "deepseek" and deepseek_key and deepseek_key != "your-deepseek-api-key-here":
        analyzer.llm_provider.set_provider('deepseek', deepseek_key)
    elif llm_provider == "openai" and openai_key and openai_key != "your-openai-api-key-here":
        analyzer.llm_provider.set_provider('openai', openai_key)
    elif deepseek_key and deepseek_key != "your-deepseek-api-key-here":
        # Default to DeepSeek if key available
        analyzer.llm_provider.set_provider('deepseek', deepseek_key)
    elif openai_key and openai_key != "your-openai-api-key-here":
        # Fallback to OpenAI
        analyzer.llm_provider.set_provider('openai', openai_key)

    if finnhub_key and finnhub_key != "your-finnhub-api-key-here":
        analyzer.data_fetcher.initialize_finnhub(finnhub_key)


# Initialize environment
setup_environment()

# Initialize database manager at startup
print("🔍 初始化数据库管理器...")
try:
    from database import get_database_manager

    db_manager = get_database_manager()
    print(f"✅ 数据库管理器初始化成功")
    print(f"📁 数据库路径: {db_manager.db_path}")

    # 检查数据库文件是否存在
    import os

    if os.path.exists(db_manager.db_path):
        file_size = os.path.getsize(db_manager.db_path)
        print(f"📊 数据库文件大小: {file_size} 字节")
    else:
        print("⚠️  数据库文件尚未创建，将在第一次API调用时创建")

except Exception as e:
    print(f"❌ 数据库管理器初始化失败: {e}")
    import traceback

    traceback.print_exc()


# Flask routes remain unchanged, only modify API key related endpoints
@app.route('/api/update-api-key', methods=['POST'])
def update_api_key():
    """API endpoint to update LLM API key with provider support."""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        provider = data.get('provider', 'openai')

        if not api_key:
            return jsonify({"error": "API key cannot be empty"})

        # 验证API密钥
        validation = analyzer.llm_provider.validate_api_key(provider, api_key)
        if not validation["valid"]:
            return jsonify({"error": validation["error"]})

        # 设置提供商和API密钥
        analyzer.llm_provider.set_provider(provider, api_key)

        # 刷新交易图的LLM
        analyzer.trading_graph.refresh_llms()

        return jsonify({"success": True,
                        "message": f"{analyzer.llm_provider.providers[provider]['name']} API key updated successfully"})

    except Exception as e:
        return jsonify({"error": safe_str(e)})


@app.route('/api/validate-api-key', methods=['POST'])
def validate_api_key():
    """API endpoint to validate API key for a specific provider."""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        provider = data.get('provider', 'openai')

        if not api_key:
            return jsonify({"error": "API key cannot be empty"})

        validation = analyzer.llm_provider.validate_api_key(provider, api_key)
        return jsonify(validation)

    except Exception as e:
        return jsonify({"valid": False, "error": safe_str(e)})


@app.route('/api/get-api-key-status')
def get_api_key_status():
    """API endpoint to check API key status for all providers."""
    try:
        openai_key = os.environ.get("OPENAI_API_KEY")
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")

        result = {
            'has_openai_key': False,
            'has_deepseek_key': False,
            'masked_openai_key': '',
            'masked_deepseek_key': ''
        }

        if openai_key and openai_key != "your-openai-api-key-here":
            masked_key = openai_key[:8] + '*' * (len(openai_key) - 12) + openai_key[-4:] if len(
                openai_key) > 12 else '***'
            result['has_openai_key'] = True
            result['masked_openai_key'] = masked_key

        if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
            masked_key = deepseek_key[:8] + '*' * (len(deepseek_key) - 12) + deepseek_key[-4:] if len(
                deepseek_key) > 12 else '***'
            result['has_deepseek_key'] = True
            result['masked_deepseek_key'] = masked_key

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": safe_str(e)})


# 新增：存储和获取分析结果的API
@app.route('/api/store-results', methods=['POST'])
def store_results():
    """存储分析结果并返回结果ID"""
    try:
        data = request.get_json()
        results = data.get('results')

        if not results:
            return jsonify({"error": "No results provided"}), 400

        # 生成唯一的结果ID
        import uuid
        result_id = str(uuid.uuid4())

        # 存储结果到内存（生产环境应该使用Redis等持久化存储）
        analysis_results_store[result_id] = {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        # 同时存储到session
        session['last_analysis_id'] = result_id

        # 清理过期结果（24小时前的）
        cleanup_expired_results()

        return jsonify({"success": True, "result_id": result_id})

    except Exception as e:
        return jsonify({"error": safe_str(e)}), 500


@app.route('/api/get-results/<result_id>')
def get_results(result_id):
    """根据结果ID获取分析结果"""
    try:
        # 从内存存储获取结果
        if result_id in analysis_results_store:
            results = analysis_results_store[result_id]['results']
            return jsonify({"success": True, "results": results})
        else:
            return jsonify({"error": "Results not found or expired"}), 404

    except Exception as e:
        return jsonify({"error": safe_str(e)}), 500


def cleanup_expired_results():
    """清理过期的分析结果（24小时前）"""
    try:
        current_time = datetime.now()
        expired_keys = []

        for result_id, data in analysis_results_store.items():
            result_time = datetime.fromisoformat(data['timestamp'])
            if (current_time - result_time).total_seconds() > 24 * 3600:  # 24小时
                expired_keys.append(result_id)

        for key in expired_keys:
            del analysis_results_store[key]

        if expired_keys:
            print(f"清理了 {len(expired_keys)} 个过期的分析结果")

    except Exception as e:
        print(f"清理过期结果时出错: {safe_str(e)}")


# Keep other routes unchanged
@app.route('/')
def index():
    return render_template('demo_new.html')


@app.route('/QuantAgent')
def QuantAgent():
    return render_template('demo_new.html')


@app.route('/output')
def output():
    """Display analysis results page"""
    try:
        # 优先使用result_id参数
        result_id = request.args.get('result_id')
        session_result_id = session.get('last_analysis_id')

        if result_id:
            # 根据result_id从存储中获取结果
            if result_id in analysis_results_store:
                results = analysis_results_store[result_id]['results']
                print(f"使用存储的结果ID: {result_id}")
            else:
                # 如果找不到结果，返回错误
                results = {
                    "success": False,
                    "error": f"Results not found for ID: {result_id}",
                    "asset_name": "Unknown",
                    "timeframe": "Unknown",
                    "data_length": 0,
                    "cache_info": {"is_cached": False}
                }
        elif session_result_id:
            # 使用session中的结果ID
            if session_result_id in analysis_results_store:
                results = analysis_results_store[session_result_id]['results']
                print(f"使用session中的结果ID: {session_result_id}")
            else:
                results = get_default_results()
        else:
            # 向后兼容：尝试使用results参数（但限制长度）
            results_param = request.args.get('results')
            if results_param and len(results_param) < 2000:  # 限制长度避免414错误
                import urllib.parse
                try:
                    decoded_results = urllib.parse.unquote(results_param, encoding='utf-8')
                    if isinstance(decoded_results, str):
                        decoded_results = decoded_results.encode('utf-8', errors='replace').decode('utf-8')

                    results = json.loads(decoded_results)
                    print(f"使用URL参数结果（长度: {len(results_param)}）")
                except Exception as decode_error:
                    print(f"URL decode error: {safe_str(decode_error)}")
                    results = get_default_results()
            else:
                # 如果URL参数太长或不存在，使用默认结果
                results = get_default_results()

        # 确保所有字符串字段都是安全的
        results = sanitize_results(results)

        return render_template('output.html', results=results)

    except Exception as e:
        # If there's an error parsing results, show error page
        error_results = {
            "success": False,
            "error": f"Error loading results: {safe_str(e)}",
            "asset_name": "Unknown",
            "timeframe": "Unknown",
            "data_length": 0,
            "cache_info": {"is_cached": False}
        }
        return render_template('output.html', results=error_results)


def get_default_results():
    """获取默认结果"""
    return {
        "success": True,
        "asset_name": "BTC",
        "timeframe": "1h",
        "data_length": 1247,
        "technical_indicators": "No analysis data available",
        "pattern_analysis": "No pattern analysis available",
        "trend_analysis": "No trend analysis available",
        "final_decision": {
            "decision": "HOLD",
            "risk_reward_ratio": "1:1",
            "forecast_horizon": "24 hours",
            "justification": "No analysis data available"
        },
        "cache_info": {"is_cached": False}
    }


def sanitize_results(results):
    """确保结果中的所有字符串都是安全的"""
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, str):
                results[key] = safe_str(value)
            elif isinstance(value, dict):
                results[key] = sanitize_results(value)
            elif isinstance(value, list):
                results[key] = [sanitize_results(item) if isinstance(item, dict) else safe_str(item) if isinstance(item,
                                                                                                                   str) else item
                                for item in value]
    return results


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        asset = data.get('asset')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        start_time = data.get('start_time', '00:00')
        end_date = data.get('end_date')
        end_time = data.get('end_time', '23:59')
        redirect_to_output = data.get('redirect_to_output', False)
        generate_charts = data.get('generate_charts', False)  # 新增参数，默认关闭图表生成
        trading_strategy = data.get('trading_strategy', 'high_frequency')  # 新增交易策略参数，默认高频交易
        session_id = data.get('session_id')  # 新增：接收前端传递的session_id

        # 添加日志打印，确认策略参数是否正确传递
        print(f"[DEBUG] 收到的交易策略参数: {trading_strategy}")

        # Validate required parameters
        if not asset or not timeframe or not start_date or not end_date:
            return jsonify({"error": "Missing required parameters"})

        # Create datetime objects
        try:
            start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
        except ValueError:
            return jsonify({"error": "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."})

        # 首先检查数据库中是否存在相同查询条件的分析结果（24小时内）
        print(f"🔍 检查数据库缓存...")
        print(f"   📊 查询条件: {asset} {timeframe} {start_date}~{end_date} {trading_strategy}")
        existing_analysis = db_manager.check_existing_analysis(
            asset=asset,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,
            end_time=end_time,
            trading_strategy=trading_strategy,
            max_hours_old=24
        )

        if existing_analysis:
            # 如果找到缓存结果，直接返回
            print(f"✅ 使用缓存的分析结果，跳过API调用")

            # 从缓存结果中提取数据
            result_details = existing_analysis.get('result_details', {})
            result_summary = existing_analysis.get('result_summary', '')

            # 构建返回结果
            formatted_results = {
                "success": True,
                "asset_name": asset,
                "timeframe": timeframe,
                "data_length": result_details.get('data_length', 0),
                "technical_indicators": result_details.get('technical_indicators', ''),
                "pattern_analysis": result_details.get('pattern_analysis', ''),
                "trend_analysis": result_details.get('trend_analysis', ''),
                "pattern_chart": result_details.get('pattern_chart', ''),
                "trend_chart": result_details.get('trend_chart', ''),
                "pattern_image_filename": result_details.get('pattern_image_filename', ''),
                "trend_image_filename": result_details.get('trend_image_filename', ''),
                "final_decision": result_details.get('final_decision', {}),
                "cached": True,  # 标记为缓存结果
                "cache_id": existing_analysis['id'],
                "cache_timestamp": existing_analysis['created_at']
            }

            if redirect_to_output:
                # 新方法：存储结果并返回结果ID
                store_response = store_results_in_memory(formatted_results)
                if store_response['success']:
                    result_id = store_response['result_id']
                    redirect_url = f"/output?result_id={result_id}"
                    return jsonify({"redirect": redirect_url})
                else:
                    # 如果存储失败，返回结果直接
                    return jsonify(formatted_results)
            else:
                return jsonify(formatted_results)

        # 如果没有找到缓存结果，继续执行原有的分析流程
        print(f"🔍 未找到缓存，开始执行新的分析...")

        # Use new data fetching method
        df = analyzer.fetch_market_data(asset, timeframe, start_dt, end_dt)
        if df.empty:
            return jsonify({"error": "Unable to fetch data, please check the code or try other data sources"})

        display_name = analyzer.asset_mapping.get(asset, asset)
        results = analyzer.run_analysis(df, display_name, timeframe, generate_charts,
                                        trading_strategy)  # 传递generate_charts和trading_strategy参数
        formatted_results = analyzer.extract_analysis_results(results)

        # 保存分析结果到数据库
        try:
            history_id = db_manager.save_analysis_history(
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                generate_charts=generate_charts,
                trading_strategy=trading_strategy,
                result_summary=f"{asset} {timeframe} 分析结果",
                result_details=formatted_results,
                status='completed',
                session_id=session_id,  # 使用前端传递的session_id
                user_ip=request.remote_addr
            )
            print(f"✅ 分析结果已保存到数据库，ID: {history_id}")
            print(f"   📊 使用的session_id: {session_id}")
        except Exception as e:
            print(f"⚠️ 保存分析结果到数据库失败: {safe_str(e)}")

        if redirect_to_output:
            # 新方法：存储结果并返回结果ID
            store_response = store_results_in_memory(formatted_results)
            if store_response['success']:
                result_id = store_response['result_id']
                redirect_url = f"/output?result_id={result_id}"
                return jsonify({"redirect": redirect_url})
            else:
                # 如果存储失败，返回结果直接
                return jsonify(formatted_results)
        else:
            return jsonify(formatted_results)

    except Exception as e:
        error_msg = safe_str(e)
        print(f"Analysis error: {error_msg}")
        return jsonify({"error": error_msg})


def store_results_in_memory(results):
    """将结果存储到内存并返回结果ID"""
    try:
        import uuid
        result_id = str(uuid.uuid4())

        # 存储结果到内存
        analysis_results_store[result_id] = {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        # 同时存储到session
        session['last_analysis_id'] = result_id

        # 清理过期结果
        cleanup_expired_results()

        return {"success": True, "result_id": result_id}

    except Exception as e:
        print(f"存储结果到内存失败: {safe_str(e)}")
        return {"success": False, "error": safe_str(e)}


# 历史记录API端点
@app.route('/api/history/save', methods=['POST'])
def save_analysis_history():
    """保存分析历史记录"""
    try:
        data = request.get_json()

        # 获取数据库管理器
        from database import get_database_manager
        db_manager = get_database_manager()

        # 保存历史记录
        history_id = db_manager.save_analysis_history(
            asset=data.get('asset'),
            timeframe=data.get('timeframe'),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            use_current_time=data.get('use_current_time', False),
            generate_charts=data.get('generate_charts', False),
            trading_strategy=data.get('trading_strategy'),
            analysis_params=data.get('analysis_params'),
            result_summary=data.get('result_summary'),
            result_details=data.get('result_details'),
            status=data.get('status', 'pending'),
            error_message=data.get('error_message'),
            session_id=data.get('session_id'),
            user_ip=request.remote_addr
        )

        return jsonify({"success": True, "history_id": history_id})

    except Exception as e:
        error_msg = safe_str(e)
        print(f"保存历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500


@app.route('/api/history/update', methods=['POST'])
def update_analysis_history():
    """更新分析历史记录"""
    try:
        data = request.get_json()
        history_id = data.get('history_id')

        if not history_id:
            return jsonify({"success": False, "error": "Missing history_id"}), 400

        # 获取数据库管理器
        from database import get_database_manager
        db_manager = get_database_manager()

        # 更新历史记录
        success = db_manager.update_analysis_history(
            history_id=history_id,
            result_summary=data.get('result_summary'),
            result_details=data.get('result_details'),
            status=data.get('status'),
            error_message=data.get('error_message')
        )

        return jsonify({"success": success})

    except Exception as e:
        error_msg = safe_str(e)
        print(f"更新历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500


@app.route('/api/history/list', methods=['GET'])
def get_analysis_history():
    """获取分析历史记录列表"""
    try:
        # 获取查询参数
        limit = request.args.get('limit', 50, type=int)
        asset = request.args.get('asset')
        timeframe = request.args.get('timeframe')
        status = request.args.get('status')
        days_back = request.args.get('days_back', 30, type=int)

        # 获取数据库管理器
        from database import get_database_manager
        db_manager = get_database_manager()

        # 获取历史记录 - 移除session_id过滤，允许跨session查看所有记录
        history_list = db_manager.get_analysis_history_list(
            limit=limit,
            asset=asset,
            timeframe=timeframe,
            status=status,
            days_back=days_back
        )

        return jsonify({"success": True, "history": history_list})

    except Exception as e:
        error_msg = safe_str(e)
        print(f"获取历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500


@app.route('/api/history/<int:history_id>', methods=['GET'])
def get_analysis_history_by_id(history_id):
    """根据ID获取分析历史记录详情"""
    try:
        # 获取数据库管理器
        from database import get_database_manager
        db_manager = get_database_manager()

        # 获取历史记录详情
        history_record = db_manager.get_analysis_history_by_id(history_id)

        if history_record:
            return jsonify({"success": True, "record": history_record})
        else:
            return jsonify({"success": False, "error": "Record not found"}), 404

    except Exception as e:
        error_msg = safe_str(e)
        print(f"获取历史记录详情失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500


@app.route('/api/history/<int:history_id>', methods=['DELETE'])
def delete_analysis_history(history_id):
    """删除分析历史记录"""
    try:
        # 获取数据库管理器
        from database import get_database_manager
        db_manager = get_database_manager()

        # 删除历史记录
        success = db_manager.delete_analysis_history(history_id)

        return jsonify({"success": success})

    except Exception as e:
        error_msg = safe_str(e)
        print(f"删除历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500


@app.route('/api/history/clear', methods=['POST'])
def clear_analysis_history():
    """清理分析历史记录"""
    try:
        data = request.get_json() or {}
        days_older_than = data.get('days_older_than')

        # 获取数据库管理器
        from database import get_database_manager
        db_manager = get_database_manager()

        # 清理历史记录
        deleted_count = db_manager.clear_analysis_history(days_older_than)

        return jsonify({"success": True, "deleted_count": deleted_count})

    except Exception as e:
        error_msg = safe_str(e)
        print(f"清理历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500


# 添加缺失的静态资源和API路由
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets"""
    try:
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        return send_file(os.path.join(assets_dir, filename))
    except Exception as e:
        return jsonify({"error": f"Asset not found: {safe_str(e)}"}), 404


@app.route('/api/custom-assets', methods=['GET'])
def custom_assets():
    """API endpoint to get custom assets"""
    try:
        custom_assets = analyzer.load_custom_assets()
        return jsonify(custom_assets)
    except Exception as e:
        return jsonify({"error": safe_str(e)}), 500


@app.route('/api/images/<image_type>')
def get_image(image_type):
    """API endpoint to serve analysis images"""
    try:
        # 根据图片类型返回相应的图片文件
        image_dir = os.path.join(os.path.dirname(__file__), 'data', 'images')

        if image_type == 'pattern':
            # 查找最新的pattern图片
            pattern_files = [f for f in os.listdir(image_dir) if f.startswith('pattern_') and f.endswith('.png')]
            if pattern_files:
                latest_file = max(pattern_files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
                return send_file(os.path.join(image_dir, latest_file))
            else:
                # 如果没有找到分析图片，返回占位图
                placeholder_path = os.path.join(image_dir, 'pattern_placeholder.svg')
                if os.path.exists(placeholder_path):
                    return send_file(placeholder_path)
        elif image_type == 'trend':
            # 查找最新的trend图片
            trend_files = [f for f in os.listdir(image_dir) if f.startswith('trend_') and f.endswith('.png')]
            if trend_files:
                latest_file = max(trend_files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
                return send_file(os.path.join(image_dir, latest_file))
            else:
                # 如果没有找到分析图片，返回占位图
                placeholder_path = os.path.join(image_dir, 'trend_placeholder.svg')
                if os.path.exists(placeholder_path):
                    return send_file(placeholder_path)

        # 如果没有找到图片，返回404
        return jsonify({"error": f"Image not found: {image_type}"}), 404

    except Exception as e:
        return jsonify({"error": safe_str(e)}), 404


# Other helper routes remain unchanged
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ElonQuantAgent Web Interface')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()

    app.run(debug=True, host=args.host, port=args.port)
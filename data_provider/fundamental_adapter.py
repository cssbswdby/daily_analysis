# -*- coding: utf-8 -*-
"""
AkShare fundamental adapter (fail-open).

This adapter intentionally uses capability probing against multiple AkShare
endpoint candidates. It should never raise to caller; partial data is allowed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_DIVIDEND_KEYWORD_MAP: Dict[str, List[str]] = {
    "per_share": [
        "每股派息",
        "每股现金红利",
        "每股分红",
        "每股派现",
        "派现(元/股)",
        "派息(元/股)",
        "税前派息(元/股)",
        "现金分红(税前)",
    ],
    "plan_text": [
        "分配方案",
        "分红方案",
        "实施方案",
        "派息方案",
        "方案",
        "预案",
        "方案说明",
    ],
    "ex_dividend_date": ["除权除息日", "除息日", "除权日", "除权除息", "除息日期"],
    "record_date": ["股权登记日", "登记日"],
    "announce_date": ["公告日期", "公告日", "实施公告日", "预案公告日"],
    "report_date": ["报告期", "报告日期", "截止日期", "统计截止日期"],
}


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float conversion."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    try:
        return parsed.to_pydatetime()
    except Exception:
        return None


def _normalize_code(raw: Any) -> str:
    s = _safe_str(raw).upper()
    if "." in s:
        s = s.split(".", 1)[0]
    s = re.sub(r"^(SH|SZ|BJ)", "", s)
    return s


def _pick_by_keywords(row: pd.Series, keywords: List[str]) -> Optional[Any]:
    """
    Return first non-empty row value whose column name contains any keyword.
    """
    for col in row.index:
        col_s = str(col)
        if any(k in col_s for k in keywords):
            val = row.get(col)
            if val is not None and str(val).strip() not in ("", "-", "nan", "None"):
                return val
    return None


def _parse_dividend_plan_to_per_share(plan_text: str) -> Optional[float]:
    """Parse per-share cash dividend from Chinese plan text."""
    text = _safe_str(plan_text)
    if not text:
        return None

    for pattern in (
        r"(?:每)?\s*10\s*股?\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元",
        r"10\s*派\s*([0-9]+(?:\.[0-9]+)?)\s*元",
    ):
        match = re.search(pattern, text)
        if match:
            parsed = _safe_float(match.group(1))
            if parsed is not None and parsed > 0:
                return parsed / 10.0

    match_per_share = re.search(r"每\s*股\s*派(?:发)?\s*([0-9]+(?:\.[0-9]+)?)\s*元", text)
    if match_per_share:
        parsed = _safe_float(match_per_share.group(1))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _extract_cash_dividend_per_share(row: pd.Series) -> Optional[float]:
    """Extract pre-tax cash dividend per share from a row."""
    plan_text = _safe_str(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["plan_text"]))
    # Keep pre-tax semantics; skip explicit after-tax plans unless pre-tax marker exists.
    if "税后" in plan_text and "税前" not in plan_text and "含税" not in plan_text:
        return None

    direct = _safe_float(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["per_share"]))
    if direct is not None and direct > 0:
        return direct
    return _parse_dividend_plan_to_per_share(plan_text)


def _filter_rows_by_code(df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "symbol", "ts_code"))]
    if not code_cols:
        return df

    target = _normalize_code(stock_code)
    for col in code_cols:
        try:
            series = df[col].astype(str).map(_normalize_code)
            filtered = df[series == target]
            if not filtered.empty:
                return filtered
        except Exception:
            continue
    return pd.DataFrame()


def _normalize_report_date(value: Any) -> Optional[str]:
    parsed = _safe_datetime(value)
    return parsed.date().isoformat() if parsed else None


def _build_dividend_payload(
    dividend_df: pd.DataFrame,
    stock_code: str,
    max_events: int = 5,
) -> Dict[str, Any]:
    work_df = _filter_rows_by_code(dividend_df, stock_code)
    if work_df.empty:
        return {}

    now_date = datetime.now().date()
    ttm_start_date = now_date - timedelta(days=365)
    dedupe_keys = set()
    events: List[Dict[str, Any]] = []

    for _, row in work_df.iterrows():
        if not isinstance(row, pd.Series):
            continue
        ex_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["ex_dividend_date"]))
        record_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["record_date"]))
        announce_dt = _safe_datetime(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["announce_date"]))
        event_dt = ex_dt or record_dt or announce_dt
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if event_date > now_date:
            continue

        per_share = _extract_cash_dividend_per_share(row)
        if per_share is None or per_share <= 0:
            continue

        dedupe_key = (event_date.isoformat(), round(per_share, 6))
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)

        events.append(
            {
                "event_date": event_date.isoformat(),
                "ex_dividend_date": ex_dt.date().isoformat() if ex_dt else None,
                "record_date": record_dt.date().isoformat() if record_dt else None,
                "announcement_date": announce_dt.date().isoformat() if announce_dt else None,
                "cash_dividend_per_share": round(per_share, 6),
                "is_pre_tax": True,
            }
        )

    if not events:
        return {}

    events.sort(key=lambda item: item.get("event_date") or "", reverse=True)
    ttm_events: List[Dict[str, Any]] = []
    for item in events:
        event_dt = _safe_datetime(item.get("event_date"))
        if event_dt is None:
            continue
        event_date = event_dt.date()
        if ttm_start_date <= event_date <= now_date:
            ttm_events.append(item)

    return {
        "events": events[:max(1, max_events)],
        "ttm_event_count": len(ttm_events),
        "ttm_cash_dividend_per_share": (
            round(sum(float(item.get("cash_dividend_per_share") or 0.0) for item in ttm_events), 6)
            if ttm_events else None
        ),
        "coverage": "cash_dividend_pre_tax",
        "as_of": now_date.isoformat(),
    }


def _extract_latest_row(df: pd.DataFrame, stock_code: str) -> Optional[pd.Series]:
    """
    Select the most relevant row for the given stock.
    """
    if df is None or df.empty:
        return None

    code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码", "ts_code", "symbol"))]
    target = _normalize_code(stock_code)
    if code_cols:
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                matched = df[series == target]
                if not matched.empty:
                    return matched.iloc[0]
            except Exception:
                continue
        return None

    # Fallback: use latest row
    return df.iloc[0]


class AkshareFundamentalAdapter:
    """AkShare adapter for fundamentals, capital flow and dragon-tiger signals."""

    def _tencent_tick_capital_flow(self, stock_code: str) -> Tuple[Dict[str, Any], List[str]]:
        """Compute capital flow from Tencent tick-by-tick transaction data (bypasses East Money push2)."""
        errors: List[str] = []
        try:
            import akshare as ak

            code = stock_code.strip()
            if code.startswith("6"):
                symbol = f"sh{code}"
            elif code.startswith(("0", "3")):
                symbol = f"sz{code}"
            else:
                symbol = f"sh{code}"

            df = ak.stock_zh_a_tick_tx_js(symbol=symbol)
            if df is None or df.empty:
                return {}, ["tencent_tick:no_data"]

            df["amount"] = df["成交金额"].astype(float)
            df["buy"] = df["性质"].astype(str).str.contains("买")
            df["sell"] = df["性质"].astype(str).str.contains("卖")

            # Classification: 超大单>100万, 大单20-100万, 中单4-20万, 小单<4万
            categories = [
                ("super_large", df[df["amount"] > 1000000]),
                ("large", df[(df["amount"] >= 200000) & (df["amount"] <= 1000000)]),
                ("medium", df[(df["amount"] >= 40000) & (df["amount"] < 200000)]),
                ("small", df[df["amount"] < 40000]),
            ]

            breakdown = {}
            for name, sub in categories:
                buy_amt = sub[sub["buy"]]["amount"].sum()
                sell_amt = sub[sub["sell"]]["amount"].sum()
                breakdown[name] = {
                    "count": len(sub),
                    "buy": float(buy_amt),
                    "sell": float(sell_amt),
                    "net": float(buy_amt - sell_amt),
                }

            main_net = breakdown["super_large"]["net"] + breakdown["large"]["net"]

            # Top 10 largest trades
            top_df = df.nlargest(10, "amount")
            top_trades = []
            for _, row in top_df.iterrows():
                side = "买盘" if row["buy"] else ("卖盘" if row["sell"] else "中性")
                top_trades.append({
                    "time": str(row.get("成交时间", "")),
                    "price": float(row.get("成交价格", 0)),
                    "volume": int(row.get("成交量", 0)),
                    "amount": float(row.get("成交金额", 0)),
                    "side": side,
                })

            return {
                "main_net_inflow": float(main_net) if main_net else 0.0,
                "inflow_5d": None,
                "inflow_10d": None,
                "breakdown": breakdown,
                "top_trades": top_trades,
            }, ["tencent_tick:ok"]
        except Exception as exc:
            return {}, [f"tencent_tick:{type(exc).__name__}:{str(exc)[:100]}"]

    def _sina_sector_flow(self, top_n: int = 5) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Fetch sector fund flow rankings from Sina Finance (bypasses East Money)."""
        errors: List[str] = []
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}

            # Top inflow sectors
            r = requests.get(
                "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_bkzj_bk",
                params={"page": "1", "num": str(top_n), "sort": "netamount", "asc": "0", "fenlei": "1"},
                headers=headers, timeout=8,
            )
            top_data = r.json() if r.text.strip() else []
            top = [{"name": s.get("name", ""), "net_inflow": _safe_float(s.get("netamount"))} for s in (top_data or []) if s.get("name")]

            # Top outflow sectors
            r2 = requests.get(
                "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_bkzj_bk",
                params={"page": "1", "num": str(top_n), "sort": "netamount", "asc": "1", "fenlei": "1"},
                headers=headers, timeout=8,
            )
            bottom_data = r2.json() if r2.text.strip() else []
            bottom = [{"name": s.get("name", ""), "net_inflow": _safe_float(s.get("netamount"))} for s in (bottom_data or []) if s.get("name")]

            return top, bottom, ["sina_sector:ok"]
        except Exception as exc:
            return [], [], [f"sina_sector:{type(exc).__name__}:{str(exc)[:80]}"]

    def _playwright_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """Playwright-based capital flow fallback using real Chrome browser to bypass TLS fingerprinting."""
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }
        try:
            from playwright.sync_api import sync_playwright
            import json as _json

            code = stock_code.strip()
            if code.startswith("6"):
                secid = f"1.{code}"
            elif code.startswith(("0", "3")):
                secid = f"0.{code}"
            else:
                secid = f"1.{code}"

            with sync_playwright() as p:
                browser = p.chromium.launch(channel="chrome", headless=True)
                page = browser.new_page()
                page.goto("https://data.eastmoney.com/", timeout=20000, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)

                # Fetch individual stock fund flow (last 10 trading days)
                stock_raw = None
                for _attempt in range(3):
                    try:
                        stock_raw = page.evaluate('''async (secid) => {
                            const url = 'https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get';
                            const params = new URLSearchParams({
                                'secid': secid, 'lmt': '10',
                                'fields1': 'f1,f2,f3,f7',
                                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65'
                            });
                            const r = await fetch(url + '?' + params.toString());
                            return await r.text();
                        }''', secid)
                        if stock_raw and len(stock_raw) > 10:
                            break
                    except Exception:
                        page.wait_for_timeout(2000)

                stock_data = _json.loads(stock_raw)
                if stock_data.get("data") and stock_data["data"].get("klines"):
                    klines = stock_data["data"]["klines"]
                    main_nets = []
                    for kline in klines:
                        parts = kline.split(",")
                        if len(parts) >= 2:
                            main_nets.append(_safe_float(parts[1]))
                    if main_nets:
                        result["stock_flow"] = {
                            "main_net_inflow": main_nets[-1],
                            "inflow_5d": sum(main_nets[-5:]) if len(main_nets) >= 1 else None,
                            "inflow_10d": sum(main_nets[-10:]) if len(main_nets) >= 1 else None,
                        }
                        result["source_chain"].append("capital_stock:playwright")

                # Fetch sector fund flow rankings (top N by net inflow)
                sector_raw = page.evaluate('''async () => {
                    const url = 'https://push2.eastmoney.com/api/qt/clist/get';
                    const params = new URLSearchParams({
                        'fid': 'f62', 'po': '1', 'pz': '10', 'pn': '1', 'np': '1',
                        'fltt': '2', 'invt': '2', 'fs': 'm:90 t:2',
                        'fields': 'f12,f14,f62'
                    });
                    const r = await fetch(url + '?' + params.toString());
                    return await r.text();
                }''')

                sector_data = _json.loads(sector_raw)
                if sector_data.get("data") and sector_data["data"].get("diff"):
                    sectors = sector_data["data"]["diff"]
                    top = [{"name": s.get("f14", ""), "net_inflow": _safe_float(s.get("f62"))} for s in sectors[:top_n] if s.get("f14")]
                    bottom_raw = page.evaluate('''async () => {
                        const url = 'https://push2.eastmoney.com/api/qt/clist/get';
                        const params = new URLSearchParams({
                            'fid': 'f62', 'po': '0', 'pz': '10', 'pn': '1', 'np': '1',
                            'fltt': '2', 'invt': '2', 'fs': 'm:90 t:2',
                            'fields': 'f12,f14,f62'
                        });
                        const r = await fetch(url + '?' + params.toString());
                        return await r.text();
                    }''')
                    bottom_data = _json.loads(bottom_raw)
                    bottom_sectors = bottom_data.get("data", {}).get("diff", []) if bottom_data.get("data") else []
                    bottom = [{"name": s.get("f14", ""), "net_inflow": _safe_float(s.get("f62"))} for s in bottom_sectors[:top_n] if s.get("f14")]
                    result["sector_rankings"] = {"top": top, "bottom": bottom}
                    result["source_chain"].append("capital_sector:playwright")

                browser.close()

            has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"])
            result["status"] = "partial" if has_content else "not_supported"
            if not has_content:
                result["errors"].append("playwright:no_data")
        except Exception as exc:
            result["errors"].append(f"playwright:{type(exc).__name__}:{str(exc)[:100]}")
        return result

    def _tushare_moneyflow_fallback(self, stock_code: str) -> Tuple[Dict[str, Any], List[str]]:
        """Tushare moneyflow fallback when AkShare (East Money push2) is unreachable."""
        errors: List[str] = []
        try:
            import os
            token = os.environ.get("TUSHARE_TOKEN", "")
            if not token:
                return {}, ["tushare_fallback:no_token"]

            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()

            code = stock_code.strip()
            if code.startswith("6"):
                ts_code = f"{code}.SH"
            elif code.startswith(("0", "3")):
                ts_code = f"{code}.SZ"
            elif code.startswith(("8", "4")):
                ts_code = f"{code}.BJ"
            else:
                ts_code = code

            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=20)).strftime("%Y%m%d")

            df = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return {}, ["tushare_fallback:no_data"]

            df = df.sort_values("trade_date", ascending=False)
            latest = df.iloc[0]
            net_inflow = _safe_float(latest.get("net_mf_amount"))
            inflow_5d = _safe_float(df.head(5)["net_mf_amount"].sum()) if len(df) >= 1 else None
            inflow_10d = _safe_float(df.head(10)["net_mf_amount"].sum()) if len(df) >= 1 else None

            return {
                "main_net_inflow": net_inflow,
                "inflow_5d": inflow_5d,
                "inflow_10d": inflow_10d,
            }, ["tushare_fallback:ok"]
        except Exception as exc:
            return {}, [f"tushare_fallback:{type(exc).__name__}:{str(exc)[:120]}"]

    def _call_df_candidates(
        self,
        candidates: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
        errors: List[str] = []
        try:
            import akshare as ak
        except Exception as exc:
            return None, None, [f"import_akshare:{type(exc).__name__}"]

        for func_name, kwargs in candidates:
            fn = getattr(ak, func_name, None)
            if fn is None:
                continue
            try:
                df = fn(**kwargs)
                if isinstance(df, pd.Series):
                    df = df.to_frame().T
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df, func_name, errors
            except Exception as exc:
                errors.append(f"{func_name}:{type(exc).__name__}")
                continue
        return None, None, errors

    def get_fundamental_bundle(self, stock_code: str) -> Dict[str, Any]:
        """
        Return normalized fundamental blocks from AkShare with partial tolerance.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "growth": {},
            "earnings": {},
            "institution": {},
            "source_chain": [],
            "errors": [],
        }

        # Financial indicators
        fin_df, fin_source, fin_errors = self._call_df_candidates([
            ("stock_financial_abstract", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {"symbol": stock_code}),
            ("stock_financial_analysis_indicator", {}),
        ])
        result["errors"].extend(fin_errors)
        if fin_df is not None:
            row = _extract_latest_row(fin_df, stock_code)
            if row is not None:
                revenue_yoy = _safe_float(_pick_by_keywords(row, ["营业收入同比", "营收同比", "收入同比", "同比增长"]))
                profit_yoy = _safe_float(_pick_by_keywords(row, ["净利润同比", "净利同比", "归母净利润同比"]))
                roe = _safe_float(_pick_by_keywords(row, ["净资产收益率", "ROE", "净资产收益"]))
                gross_margin = _safe_float(_pick_by_keywords(row, ["毛利率"]))
                report_date = _normalize_report_date(_pick_by_keywords(row, _DIVIDEND_KEYWORD_MAP["report_date"]))
                revenue = _safe_float(_pick_by_keywords(row, ["营业总收入", "营业收入", "营收"]))
                net_profit_parent = _safe_float(_pick_by_keywords(row, ["归母净利润", "母公司股东净利润", "净利润"]))
                operating_cash_flow = _safe_float(
                    _pick_by_keywords(row, ["经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"])
                )
                result["growth"] = {
                    "revenue_yoy": revenue_yoy,
                    "net_profit_yoy": profit_yoy,
                    "roe": roe,
                    "gross_margin": gross_margin,
                }
                financial_report_payload = {
                    "report_date": report_date,
                    "revenue": revenue,
                    "net_profit_parent": net_profit_parent,
                    "operating_cash_flow": operating_cash_flow,
                    "roe": roe,
                }
                if any(v is not None for v in financial_report_payload.values()):
                    result["earnings"]["financial_report"] = financial_report_payload
                result["source_chain"].append(f"growth:{fin_source}")

        # Earnings forecast
        forecast_df, forecast_source, forecast_errors = self._call_df_candidates([
            ("stock_yjyg_em", {"symbol": stock_code}),
            ("stock_yjyg_em", {}),
            ("stock_yjbb_em", {"symbol": stock_code}),
            ("stock_yjbb_em", {}),
        ])
        result["errors"].extend(forecast_errors)
        if forecast_df is not None:
            row = _extract_latest_row(forecast_df, stock_code)
            if row is not None:
                result["earnings"]["forecast_summary"] = _safe_str(
                    _pick_by_keywords(row, ["预告", "业绩变动", "内容", "摘要", "公告"])
                )[:200]
                result["source_chain"].append(f"earnings_forecast:{forecast_source}")

        # Earnings quick report
        quick_df, quick_source, quick_errors = self._call_df_candidates([
            ("stock_yjkb_em", {"symbol": stock_code}),
            ("stock_yjkb_em", {}),
        ])
        result["errors"].extend(quick_errors)
        if quick_df is not None:
            row = _extract_latest_row(quick_df, stock_code)
            if row is not None:
                result["earnings"]["quick_report_summary"] = _safe_str(
                    _pick_by_keywords(row, ["快报", "摘要", "公告", "说明"])
                )[:200]
                result["source_chain"].append(f"earnings_quick:{quick_source}")

        # Dividend details (cash dividend, pre-tax)
        dividend_df, dividend_source, dividend_errors = self._call_df_candidates([
            ("stock_fhps_detail_em", {"symbol": stock_code}),
            ("stock_history_dividend_detail", {"symbol": stock_code, "indicator": "分红", "date": ""}),
            ("stock_dividend_cninfo", {"symbol": stock_code}),
        ])
        result["errors"].extend(dividend_errors)
        if dividend_df is not None:
            dividend_payload = _build_dividend_payload(dividend_df, stock_code, max_events=5)
            if dividend_payload:
                result["earnings"]["dividend"] = dividend_payload
                result["source_chain"].append(f"dividend:{dividend_source}")

        # Institution / top shareholders
        inst_df, inst_source, inst_errors = self._call_df_candidates([
            ("stock_institute_hold", {}),
            ("stock_institute_recommend", {}),
        ])
        result["errors"].extend(inst_errors)
        if inst_df is not None:
            row = _extract_latest_row(inst_df, stock_code)
            if row is not None:
                inst_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "变动", "持股变化"]))
                result["institution"]["institution_holding_change"] = inst_change
                result["source_chain"].append(f"institution:{inst_source}")

        top10_df, top10_source, top10_errors = self._call_df_candidates([
            ("stock_gdfx_top_10_em", {"symbol": stock_code}),
            ("stock_gdfx_top_10_em", {}),
            ("stock_zh_a_gdhs_detail_em", {"symbol": stock_code}),
            ("stock_zh_a_gdhs_detail_em", {}),
        ])
        result["errors"].extend(top10_errors)
        if top10_df is not None:
            row = _extract_latest_row(top10_df, stock_code)
            if row is not None:
                holder_change = _safe_float(_pick_by_keywords(row, ["增减", "变化", "持股变化", "变动"]))
                result["institution"]["top10_holder_change"] = holder_change
                result["source_chain"].append(f"top10:{top10_source}")

        has_content = bool(result["growth"] or result["earnings"] or result["institution"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_capital_flow(self, stock_code: str, top_n: int = 5) -> Dict[str, Any]:
        """
        Return stock + sector capital flow.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "stock_flow": {},
            "sector_rankings": {"top": [], "bottom": []},
            "source_chain": [],
            "errors": [],
        }

        # 1. Try Tencent tick data first (bypasses East Money push2 anti-bot)
        tx_flow, tx_errors = self._tencent_tick_capital_flow(stock_code)
        result["errors"].extend(tx_errors)
        if tx_flow:
            result["stock_flow"] = tx_flow
            result["source_chain"].append("capital_stock:tencent_tick")

        # 2. Try Sina for sector rankings first (bypasses East Money push2)
        sina_top, sina_bottom, sina_errors = self._sina_sector_flow(top_n)
        result["errors"].extend(sina_errors)
        if sina_top:
            result["sector_rankings"]["top"] = sina_top
            result["sector_rankings"]["bottom"] = sina_bottom
            result["source_chain"].append("capital_sector:sina")

        # 3. Fallback: AkShare for stock_flow if Tencent failed
        if not result["stock_flow"]:
            stock_df, stock_source, stock_errors = self._call_df_candidates([
                ("stock_individual_fund_flow", {"stock": stock_code}),
                ("stock_individual_fund_flow", {"symbol": stock_code}),
                ("stock_individual_fund_flow", {}),
                ("stock_main_fund_flow", {"symbol": stock_code}),
                ("stock_main_fund_flow", {}),
            ])
            result["errors"].extend(stock_errors)
            if stock_df is not None:
                row = _extract_latest_row(stock_df, stock_code)
                if row is not None:
                    net_inflow = _safe_float(_pick_by_keywords(row, ["主力净流入", "净流入", "净额"]))
                    inflow_5d = _safe_float(_pick_by_keywords(row, ["5日", "五日"]))
                    inflow_10d = _safe_float(_pick_by_keywords(row, ["10日", "十日"]))
                    result["stock_flow"] = {
                        "main_net_inflow": net_inflow,
                        "inflow_5d": inflow_5d,
                        "inflow_10d": inflow_10d,
                    }
                    result["source_chain"].append(f"capital_stock:{stock_source}")

        # 4. Fallback: Tushare if still no stock_flow
        if not result["stock_flow"]:
            ts_flow, ts_errors = self._tushare_moneyflow_fallback(stock_code)
            result["errors"].extend(ts_errors)
            if ts_flow:
                result["stock_flow"] = ts_flow
                result["source_chain"].append("capital_stock:tushare_fallback")

        # 5. Fallback: AkShare for sector rankings if Sina failed
        if not result["sector_rankings"]["top"]:
            sector_df, sector_source, sector_errors = self._call_df_candidates([
                ("stock_sector_fund_flow_rank", {}),
                ("stock_sector_fund_flow_summary", {}),
            ])
            result["errors"].extend(sector_errors)
            if sector_df is not None:
                name_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("板块", "行业", "名称", "name"))), None)
                flow_col = next((c for c in sector_df.columns if any(k in str(c) for k in ("净流入", "主力", "flow", "净额"))), None)
                if name_col and flow_col:
                    work_df = sector_df[[name_col, flow_col]].copy()
                    work_df[flow_col] = pd.to_numeric(work_df[flow_col], errors="coerce")
                    work_df = work_df.dropna(subset=[flow_col])
                    top_df = work_df.nlargest(top_n, flow_col)
                    bottom_df = work_df.nsmallest(top_n, flow_col)
                    result["sector_rankings"] = {
                        "top": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in top_df.iterrows()],
                        "bottom": [{"name": _safe_str(r[name_col]), "net_inflow": float(r[flow_col])} for _, r in bottom_df.iterrows()],
                    }
                    result["source_chain"].append(f"capital_sector:{sector_source}")

        if not result["stock_flow"] or not result["sector_rankings"]["top"]:
            pw_result = self._playwright_capital_flow(stock_code, top_n)
            if not result["stock_flow"] and pw_result["stock_flow"]:
                result["stock_flow"] = pw_result["stock_flow"]
                result["source_chain"].extend(pw_result["source_chain"])
            if not result["sector_rankings"]["top"] and pw_result["sector_rankings"]["top"]:
                result["sector_rankings"] = pw_result["sector_rankings"]
                result["source_chain"].extend(
                    [s for s in pw_result["source_chain"] if "sector" in s]
                )
            result["errors"].extend(pw_result["errors"])

        has_content = bool(result["stock_flow"] or result["sector_rankings"]["top"] or result["sector_rankings"]["bottom"])
        result["status"] = "partial" if has_content else "not_supported"
        return result

    def get_dragon_tiger_flag(self, stock_code: str, lookback_days: int = 20) -> Dict[str, Any]:
        """
        Return dragon-tiger signal in lookback window.
        """
        result: Dict[str, Any] = {
            "status": "not_supported",
            "is_on_list": False,
            "recent_count": 0,
            "latest_date": None,
            "source_chain": [],
            "errors": [],
        }

        df, source, errors = self._call_df_candidates([
            ("stock_lhb_stock_statistic_em", {}),
            ("stock_lhb_detail_em", {}),
            ("stock_lhb_jgmmtj_em", {}),
        ])
        result["errors"].extend(errors)
        if df is None:
            return result

        # Try code filter
        code_cols = [c for c in df.columns if any(k in str(c) for k in ("代码", "股票代码", "证券代码"))]
        target = _normalize_code(stock_code)
        matched = pd.DataFrame()
        for col in code_cols:
            try:
                series = df[col].astype(str).map(_normalize_code)
                cur = df[series == target]
                if not cur.empty:
                    matched = cur
                    break
            except Exception:
                continue
        if matched.empty:
            result["source_chain"].append(f"dragon_tiger:{source}")
            result["status"] = "ok" if code_cols else "partial"
            return result

        date_col = next((c for c in matched.columns if any(k in str(c) for k in ("日期", "上榜", "交易日", "time"))), None)
        parsed_dates: List[datetime] = []
        if date_col is not None:
            for val in matched[date_col].astype(str).tolist():
                try:
                    parsed_dates.append(pd.to_datetime(val).to_pydatetime())
                except Exception:
                    continue
        now = datetime.now()
        start = now - timedelta(days=max(1, lookback_days))
        recent_dates = [d for d in parsed_dates if start <= d <= now]

        result["is_on_list"] = bool(recent_dates)
        result["recent_count"] = len(recent_dates) if recent_dates else int(len(matched))
        result["latest_date"] = max(recent_dates).date().isoformat() if recent_dates else (
            max(parsed_dates).date().isoformat() if parsed_dates else None
        )
        result["status"] = "ok"
        result["source_chain"].append(f"dragon_tiger:{source}")
        return result

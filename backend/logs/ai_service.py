import json
import logging
import os

import requests


logger = logging.getLogger(__name__)

AI_ADVICE_ERROR_MESSAGE = (
    "現在AIアドバイスを取得できません。時間をおいてお試しください"
)
AI_ADVICE_TIMEOUT_MESSAGE = (
    "AI分析の応答に時間がかかっています。しばらくしてからもう一度お試しください。"
)
REQUEST_CONNECT_TIMEOUT_SECONDS = 5
REQUEST_READ_TIMEOUT_SECONDS = 45
REQUEST_TIMEOUT = (
    REQUEST_CONNECT_TIMEOUT_SECONDS,
    REQUEST_READ_TIMEOUT_SECONDS,
)

SYSTEM_PROMPT = (
    "あなたは育児記録の振り返りを支援します。単なる要約ではなく、"
    "記録の継続状況と日々のデータから分かる特徴を具体的に説明してください。"
    "入力JSONにある集計値、日別データ、比較値だけを根拠にし、"
    "事実にない数値・出来事・傾向を作らないでください。"
    "既存連携を保つため、次の6キーを持つJSONオブジェクトのみ返してください: "
    "overall_summary, item_analysis, changes_and_trends, points_to_check, "
    "observation_points, daily_suggestions。各値は日本語の文字列です。"

    "【出力内容】"
    "overall_summaryは『全体の振り返り』です。対象期間と総記録件数を簡潔に示し、"
    "記録が何日続いているか・どの項目が記録されているか、最重要な具体的特徴を先に述べます。"
    "item_analysisは『記録状況の評価』です。recording_coverageの日数・件数を使い、"
    "項目ごとの記録状況を評価します。継続している記録は肯定的かつ具体的に伝えます。"
    "記録がない項目は良否や健康状態を評価せず、未記録として扱います。"
    "changes_and_trendsは『記録から分かること』です。日別値が一定ならその安定性を、"
    "変化があれば日付と必要な数値を示して意味を説明します。比較できなくても、"
    "確認できる事実や日ごとの特徴を説明し、比較不能の説明だけで終わりません。"
    "points_to_checkとobservation_pointsは画面上で『今後の振り返りポイント』にまとめて表示されます。"
    "今回の記録に即して、次に確認すると役立つ具体的な項目を示します。"
    "一般論の繰り返しや、根拠のない問題点を作らず、必要な場合だけ記録不足や比較上の注意を簡潔に述べます。"
    "daily_suggestionsは『ひとことアドバイス』です。今回のデータに基づく肯定的な一言と、"
    "次にできることを1〜2個だけ提案します。毎回記録継続を勧める定型文にしません。"

    "【数値と比較】"
    "日ごとの値を羅列せず、特徴を説明するために必要な日付・範囲・平均・変化量のみ引用します。"
    "is_complete_dayがfalseの日は途中集計です。完了日と合計比較せず、比較から除外してください。"
    "比較期間の日数が異なる場合は入力の1日あたり平均を使い、期間合計を直接比較しません。"
    "比較に十分なデータがなければ無理に比較せず、記録日数や日別の観察可能な特徴を述べます。"
    "記録がないことを実際に出来事がなかったことと混同せず、欠損をゼロにしません。"
    "授乳量・睡眠時間・体温などの数値だけから健康・発育状態を診断せず、"
    "正常・健康・問題なしと断定したり、根拠のない基準や危険判定を作ったりしません。"
    "情報が少ないときは簡潔に、具体的な特徴が多いときは重要な発見を優先して説明します。"
    "免責説明を繰り返さず、JSON以外の前置きやMarkdownコードブロックを返しません。"
)

REPORT_SECTION_KEYS = (
    "overall_summary",
    "item_analysis",
    "changes_and_trends",
    "points_to_check",
    "observation_points",
    "daily_suggestions",
)


def generate_baby_care_advice(log_summary):
    endpoint = os.environ.get("AI_API_URL")
    api_key = os.environ.get("AI_API_KEY")
    model = os.environ.get("AI_MODEL") or "gpt-6-luna"

    if not endpoint or not api_key:
        logger.warning("AI API settings are not configured.")
        return AI_ADVICE_ERROR_MESSAGE

    summary = (
        log_summary
        if isinstance(log_summary, str)
        else json.dumps(log_summary, ensure_ascii=False, default=str)
    )
    input_size_bytes = len(summary.encode("utf-8"))

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": summary,
                "max_output_tokens": 1800,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()

    except requests.Timeout as error:
        logger.warning(
            "AI advice request timed out (%s); input_bytes=%d; "
            "connect_timeout=%ss; read_timeout=%ss.",
            type(error).__name__,
            input_size_bytes,
            REQUEST_CONNECT_TIMEOUT_SECONDS,
            REQUEST_READ_TIMEOUT_SECONDS,
        )
        return AI_ADVICE_TIMEOUT_MESSAGE

    except requests.RequestException as error:
        logger.warning(
            "AI advice request failed (%s); input_bytes=%d.",
            type(error).__name__,
            input_size_bytes,
        )
        return AI_ADVICE_ERROR_MESSAGE

    except ValueError as error:
        logger.warning(
            "AI advice API returned invalid JSON (%s).",
            type(error).__name__,
        )
        return AI_ADVICE_ERROR_MESSAGE

    # Responses APIのoutputから生成テキストを取得
    text_parts = []

    if not isinstance(payload, dict) or not isinstance(payload.get("output"), list):
        logger.error("AI advice API response has an invalid output structure.")
        return AI_ADVICE_ERROR_MESSAGE

    for item in payload["output"]:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue

        content_items = item.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                text = content.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)

    response_text = "\n".join(text_parts).strip()

    if not response_text:
        logger.error(
            "AI advice API response did not contain valid text."
        )
        return AI_ADVICE_ERROR_MESSAGE

    try:
        report = json.loads(response_text)
    except json.JSONDecodeError:
        logger.error("AI advice API response was not valid report JSON.")
        return AI_ADVICE_ERROR_MESSAGE
    if not isinstance(report, dict) or any(
        not isinstance(report.get(key), str) or not report[key].strip()
        for key in REPORT_SECTION_KEYS
    ):
        logger.error("AI advice API response did not contain all report sections.")
        return AI_ADVICE_ERROR_MESSAGE

    return {key: report[key].strip() for key in REPORT_SECTION_KEYS}

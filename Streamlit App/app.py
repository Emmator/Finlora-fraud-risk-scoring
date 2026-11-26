
# FINLORA FRAUD RISK SCORING SYSTEM
# Streamlit research prototype — single transaction scoring

from pathlib import Path
import math
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Finlora Fraud Risk Scoring",
    page_icon="🛡️",
    layout="wide"
)

ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "finlora_rf_pipeline.joblib"
)


@st.cache_resource(show_spinner="Loading Random Forest model...")
def load_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model package not found: {ARTIFACT_PATH}"
        )
    return joblib.load(ARTIFACT_PATH)


def get_category_options(preprocessor):
    """Read categorical options from the fitted encoder."""
    result = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue

        encoder = transformer

        if hasattr(transformer, "named_steps"):
            for step in transformer.named_steps.values():
                if hasattr(step, "categories_"):
                    encoder = step
                    break

        if not hasattr(encoder, "categories_"):
            continue

        for col, categories in zip(columns, encoder.categories_):
            result[str(col)] = [
                str(value)
                for value in categories
                if not pd.isna(value)
            ]

    return result


def prepare_transaction(
    amount,
    historical_average,
    velocity,
    hour,
    cross_border,
    device_status,
    currency,
    account_type,
    kyc_tier,
    merchant_category,
    channel,
    input_features
):
    """Reproduce the verified notebook feature engineering."""

    amount = float(amount)
    historical_average = float(historical_average)

    if not all(
        math.isfinite(value)
        for value in [amount, historical_average]
    ):
        raise ValueError("Amount and baseline must be finite numbers.")

    if velocity < 0:
        raise ValueError("Transaction velocity cannot be negative.")

    if not 0 <= hour <= 23:
        raise ValueError("Transaction hour must be between 0 and 23.")

    spending_ratio = (
        abs(amount) / max(abs(historical_average), 1.0)
    )

    values = {
        "amount": amount,
        "amount_to_avg_ratio": spending_ratio,
        "transaction_velocity_1h": int(velocity),
        "hour_of_day": int(hour),
        "is_cross_border": int(cross_border),
        "is_new_device": device_status,
        "device_info_missing": int(device_status == "Unknown"),
        "is_negative_amount": int(amount < 0),
        "is_negative_baseline": int(historical_average < 0),
        "currency": currency,
        "account_type": account_type,
        "kyc_tier": kyc_tier,
        "merchant_category": merchant_category,
        "channel": channel,
        "signed_log_amount": (
            np.sign(amount) * np.log1p(abs(amount))
        ),
        "log_spending_ratio": np.log1p(spending_ratio),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24)
    }

    missing = [
        feature for feature in input_features
        if feature not in values
    ]

    if missing:
        raise ValueError(
            "Missing model features: " + ", ".join(missing)
        )

    return pd.DataFrame(
        [values],
        columns=input_features
    )


# Header
st.title("🛡️ Finlora Fraud Risk Scoring System")
st.caption("Random Forest | Research prototype")


# Load package
try:
    load_start = time.perf_counter()
    package = load_model()
    load_seconds = time.perf_counter() - load_start

    model = package["model"]
    preprocessor = package["preprocessor"]
    input_features = list(package["input_features"])
    threshold = float(package["decision_threshold"])

    category_options = get_category_options(preprocessor)

except Exception as exc:
    st.error(f"Unable to load the model package: {exc}")
    st.stop()


def category_input(feature, label):
    options = category_options.get(feature, [])

    if not options:
        st.error(
            f"No fitted categories found for {feature}. "
            "Check the saved preprocessing pipeline."
        )
        st.stop()

    return st.selectbox(label, sorted(options))


st.subheader("Single Transaction Scoring")

with st.form("transaction_form"):

    left, right = st.columns(2)

    with left:
        st.markdown("#### Transaction details")

        amount_text = st.text_input(
            "Transaction amount (signed, original currency units)",
            placeholder="Example: 25000"
        )

        baseline_text = st.text_input(
            "Verified historical 30-day signed average",
            placeholder="Required — do not guess"
        )

        velocity = st.number_input(
            "Previous transactions in the last hour",
            min_value=0,
            value=0,
            step=1
        )

        hour = st.number_input(
            "Transaction hour (0–23)",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )

        cross_border = st.checkbox("Cross-border transaction")

        device_status = st.selectbox(
            "Device status",
            ["Known", "New", "Unknown"]
        )

    with right:
        st.markdown("#### Account and payment context")

        currency = category_input("currency", "Currency")
        account_type = category_input(
            "account_type", "Account type"
        )
        kyc_tier = category_input("kyc_tier", "KYC tier")
        merchant_category = category_input(
            "merchant_category", "Merchant category"
        )
        channel = category_input("channel", "Channel")

    submitted = st.form_submit_button(
        "Score Transaction",
        type="primary",
        use_container_width=True
    )


if submitted:
    try:
        if not amount_text.strip() or not baseline_text.strip():
            raise ValueError(
                "Transaction amount and verified historical "
                "average are both required."
            )

        amount = float(amount_text)
        historical_average = float(baseline_text)

        transaction = prepare_transaction(
            amount=amount,
            historical_average=historical_average,
            velocity=velocity,
            hour=hour,
            cross_border=cross_border,
            device_status=device_status,
            currency=currency,
            account_type=account_type,
            kyc_tier=kyc_tier,
            merchant_category=merchant_category,
            channel=channel,
            input_features=input_features
        )

        start = time.perf_counter()

        processed = preprocessor.transform(transaction)
        score = float(
            model.predict_proba(processed)[0, 1]
        )

        scoring_seconds = time.perf_counter() - start

        flagged = score >= threshold

        st.divider()
        st.subheader("Scoring Result")

        a, b, c = st.columns(3)
        a.metric("Model risk score", f"{score:.4f}")
        b.metric("Review threshold", f"{threshold:.2f}")
        c.metric(
            "Scoring time",
            f"{scoring_seconds * 1000:.1f} ms"
        )

        if flagged:
            st.error(
                "Review recommended — score meets "
                "the demonstration threshold."
            )
        else:
            st.success(
                "Below the demonstration review threshold. "
                "Fraud is not ruled out."
            )

        st.markdown("#### Transaction risk indicators")

        reasons = []

        if velocity > 0:
            reasons.append(
                "Previous account activity within one hour"
            )

        if transaction.iloc[0]["amount_to_avg_ratio"] > 10:
            reasons.append(
                "Spending amount exceeds 10× the supplied baseline"
            )

        if device_status == "New":
            reasons.append("New device")

        if device_status == "Unknown":
            reasons.append("Device information unavailable")

        if cross_border:
            reasons.append("Cross-border activity")

        if reasons:
            for reason in reasons:
                st.write("•", reason)
        else:
            st.info(
                "No predefined manual risk indicators triggered. "
                "The model score reflects all its input features."
            )

        with st.expander("View processed transaction inputs"):
            st.dataframe(transaction.T, use_container_width=True)

        st.caption(
            "Indicators describe transaction characteristics; "
            "they are not explanations of individual tree decisions."
        )

    except ValueError as exc:
        st.error(f"Please correct the input: {exc}")

    except Exception as exc:
        st.error(f"Transaction scoring failed: {exc}")


st.divider()
st.caption(
    f"Model: {package['model_name']} | "
    f"Input features: {len(input_features)} | "
    f"Model load call: {load_seconds:.3f}s"
)


# ============================================================
# STEP 6.5 — FRAUD REVIEW QUEUE
# ============================================================

st.divider()
st.subheader("📋 Fraud Review Queue")

st.write(
    "Upload a cleaned Finlora transaction CSV to generate "
    "a demonstration analyst review queue."
)

st.info(
    "For research testing only. Uploaded spending-ratio values "
    "may include the unresolved historical-baseline fallback."
)

queue_file = st.file_uploader(
    "Upload cleaned transaction CSV",
    type=["csv"],
    key="review_queue_upload"
)

queue_threshold = st.slider(
    "Review queue score threshold",
    min_value=0.0,
    max_value=1.0,
    value=threshold,
    step=0.01
)

if queue_file is not None:
    try:
        queue_data = pd.read_csv(queue_file)

        required = [
            feature for feature in input_features
            if feature not in [
                "signed_log_amount",
                "log_spending_ratio",
                "hour_sin",
                "hour_cos"
            ]
        ]

        missing = [
            col for col in required
            if col not in queue_data.columns
        ]

        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(missing)
            )

        if queue_data.empty:
            raise ValueError("The uploaded CSV is empty.")

        # Limit this initial prototype to 2,000 transactions
        if len(queue_data) > 2000:
            st.warning(
                "Demonstration limit: scoring only the "
                "first 2,000 uploaded transactions."
            )

        queue_data = queue_data.head(2000).copy()

        # Validate required inputs
        allowed_missing = {"is_new_device"}

        required_non_null = [
            col for col in required
            if col not in allowed_missing
        ]

        if queue_data[required_non_null].isna().any().any():
            raise ValueError(
                "Missing values found in required model "
                "inputs. Clean the file before scoring."
            )

        numeric_cols = [
            "amount",
            "amount_to_avg_ratio",
            "transaction_velocity_1h",
            "hour_of_day",
            "is_cross_border",
            "device_info_missing",
            "is_negative_amount",
            "is_negative_baseline"
        ]

        for col in numeric_cols:
            queue_data[col] = pd.to_numeric(
                queue_data[col],
                errors="raise"
            )

        if not np.isfinite(
            queue_data[numeric_cols].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Numeric fields contain invalid or infinite values."
            )

        if (
            (queue_data["amount_to_avg_ratio"] < 0).any()
            or (queue_data["transaction_velocity_1h"] < 0).any()
            or (~queue_data["hour_of_day"].between(0, 23)).any()
        ):
            raise ValueError(
                "Invalid spending ratio, velocity, or hour values."
            )

        # Match the verified device encoding
        device_values = pd.to_numeric(
            queue_data["is_new_device"],
            errors="coerce"
        )

        if not device_values.dropna().isin([0, 1]).all():
            raise ValueError(
                "Device indicator must be 0, 1, or missing."
            )

        queue_data["is_new_device"] = (
            device_values
            .map({0: "Known", 1: "New"})
            .fillna("Unknown")
        )

        # Recreate original engineered features
        queue_data["signed_log_amount"] = (
            np.sign(queue_data["amount"])
            * np.log1p(queue_data["amount"].abs())
        )

        queue_data["log_spending_ratio"] = np.log1p(
            queue_data["amount_to_avg_ratio"]
        )

        queue_data["hour_sin"] = np.sin(
            2 * np.pi * queue_data["hour_of_day"] / 24
        )

        queue_data["hour_cos"] = np.cos(
            2 * np.pi * queue_data["hour_of_day"] / 24
        )

        # Score uploaded transactions
        batch_start = time.perf_counter()

        model_inputs = queue_data[input_features].copy()
        processed_queue = preprocessor.transform(model_inputs)

        queue_data["Risk_Score"] = model.predict_proba(
            processed_queue
        )[:, 1]

        batch_seconds = time.perf_counter() - batch_start

        queue_data["Review_Status"] = np.where(
            queue_data["Risk_Score"] >= queue_threshold,
            "Review recommended",
            "Below threshold"
        )

        queue_data = queue_data.sort_values(
            "Risk_Score",
            ascending=False
        )

        flagged_count = int(
            (queue_data["Risk_Score"] >= queue_threshold).sum()
        )

        st.markdown("#### Queue Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Transactions scored", len(queue_data))
        col2.metric("Review recommendations", flagged_count)
        col3.metric(
            "Batch scoring time",
            f"{batch_seconds:.3f} s"
        )

        queue_filter = st.selectbox(
            "Filter transactions",
            [
                "All transactions",
                "Review recommended",
                "Below threshold"
            ]
        )

        if queue_filter == "All transactions":
            displayed_queue = queue_data
        else:
            displayed_queue = queue_data[
                queue_data["Review_Status"] == queue_filter
            ]

        preferred_columns = [
            "transaction_id",
            "timestamp",
            "account_id",
            "amount",
            "currency",
            "account_type",
            "kyc_tier",
            "channel",
            "merchant_category",
            "Risk_Score",
            "Review_Status"
        ]

        visible_columns = [
            col for col in preferred_columns
            if col in displayed_queue.columns
        ]

        st.dataframe(
            displayed_queue[visible_columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download scored review queue",
            data=displayed_queue[
                visible_columns
            ].to_csv(index=False),
            file_name="finlora_review_queue.csv",
            mime="text/csv"
        )

    except Exception as exc:
        st.error(f"Review queue could not be generated: {exc}")

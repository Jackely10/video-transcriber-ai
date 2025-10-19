"""Stripe Payment Integration."""

from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any, Dict, Optional

import stripe
from flask import jsonify, request

# Configure Stripe with secret key from environment.
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

PAY_PER_USE_RATE_EUR_PER_MINUTE = 0.10

PLANS: Dict[str, Dict[str, Optional[object]]] = {
    "free": {"name": "Free", "minutes": 30, "price": 0, "stripe_id": None},
    "starter": {
        "name": "Starter",
        "minutes": 100,
        "price": 9,
        "stripe_id": "price_1SInlTFtaUrvNBYYGZZ8NWLs",
    },
    "pro": {
        "name": "Pro",
        "minutes": 500,
        "price": 29,
        "stripe_id": "price_1SIoIEFtaUrvNBYY4BRIsHRY",
    },
    "business": {
        "name": "Business",
        "minutes": 2000,
        "price": 99,
        "stripe_id": "price_1SIoKnFtaUrvNBYYsHW8rICM",
    },
}

COUPONS: Dict[str, Dict[str, Any]] = {
    "EARLY20": {
        "percent_off": 20,
        "plans": None,
        "stripe_id": os.getenv("COUPON_EARLY20_ID"),
    },
    "LAUNCH30": {
        "percent_off": 30,
        "plans": {"pro"},
        "stripe_id": os.getenv("COUPON_LAUNCH30_ID"),
    },
}


def _resolve_coupon(code: str, plan_id: str) -> Optional[str]:
    if not code:
        return None
    details = COUPONS.get(code.upper())
    if not details:
        raise ValueError("Unbekannter Gutschein-Code.")
    allowed_plans = details.get("plans")
    if allowed_plans and plan_id not in allowed_plans:
        raise ValueError("Dieser Gutschein gilt nicht für den gewählten Plan.")
    coupon_id = details.get("stripe_id")
    if coupon_id:
        return coupon_id
    percent_off = details.get("percent_off")
    if not percent_off:
        raise ValueError("Ungültige Gutschein-Konfiguration.")
    try:
        coupon = stripe.Coupon.create(
            name=code.upper(),
            duration="once",
            percent_off=percent_off,
        )
    except stripe.error.StripeError as exc:  # type: ignore[attr-defined]
        raise ValueError(f"Gutschein konnte nicht aktiviert werden: {exc}") from exc
    details["stripe_id"] = coupon.id
    return coupon.id


def create_checkout_session(
    plan_id: str,
    user_email: str,
    success_url: str,
    cancel_url: str,
    coupon_code: Optional[str] = None,
):
    """Create a Stripe Checkout session for subscriptions."""
    plan = PLANS.get(plan_id)
    if not plan or plan["price"] == 0 or not plan["stripe_id"]:
        return None

    discounts = []
    applied_coupon = None
    if coupon_code:
        coupon_id = _resolve_coupon(coupon_code, plan_id)
        if coupon_id:
            discounts.append({"coupon": coupon_id})
            applied_coupon = coupon_code.upper()

    metadata = {"plan_id": plan_id, "user_email": user_email}
    if applied_coupon:
        metadata["coupon_code"] = applied_coupon

    session_kwargs: Dict[str, Any] = {
        "payment_method_types": ["card"],
        "line_items": [{"price": plan["stripe_id"], "quantity": 1}],
        "mode": "subscription",
        "success_url": success_url,
        "cancel_url": cancel_url,
        "customer_email": user_email,
        "metadata": metadata,
    }
    if discounts:
        session_kwargs["discounts"] = discounts

    session = stripe.checkout.Session.create(
        **session_kwargs,
    )
    return session


def create_pay_per_use_session(
    minutes: float, user_email: str, success_url: str, cancel_url: str
):
    """Create a Stripe Checkout session for Pay-per-Use purchases."""
    if minutes <= 0:
        raise ValueError("minutes must be positive")

    amount_cents = int(round(minutes * PAY_PER_USE_RATE_EUR_PER_MINUTE * 100))
    if amount_cents <= 0:
        raise ValueError("calculated amount must be positive")

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[
            {
                "price_data": {
                    "currency": "eur",
                    "product_data": {"name": f"{minutes} Minuten Video-Transkription"},
                    "unit_amount": amount_cents,
                },
                "quantity": 1,
            }
        ],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        customer_email=user_email,
        metadata={"type": "pay_per_use", "minutes": minutes, "user_email": user_email},
    )
    return session


def add_payment_routes(app):
    """Register payment related Flask routes on the given app instance."""

    @app.route("/api/checkout/subscription", methods=["POST"])
    def checkout_subscription():
        data = request.get_json(force=True, silent=True) or {}
        plan_id = (data.get("plan_id") or "").strip().lower()
        email = (data.get("email") or "").strip()
        coupon_code_raw = (data.get("coupon_code") or "").strip()
        coupon_code = coupon_code_raw.upper() or None
        if not email or "@" not in email:
            return jsonify({"error": "Valid email required"}), 400

        try:
            session = create_checkout_session(
                plan_id,
                email,
                success_url=f"{request.host_url.rstrip('/')}/success",
                cancel_url=f"{request.host_url.rstrip('/')}/pricing.html",
                coupon_code=coupon_code,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except stripe.error.StripeError as exc:  # type: ignore[attr-defined]
            return jsonify({"error": str(exc)}), 400
        if not session:
            return jsonify({"error": "Invalid plan"}), 400
        return jsonify({"checkout_url": session.url})

    @app.route("/api/checkout/pay-per-use", methods=["POST"])
    def checkout_pay_per_use():
        data = request.get_json(force=True, silent=True) or {}
        email = (data.get("email") or "").strip()
        minutes = float(data.get("minutes") or 0)
        if not email or "@" not in email:
            return jsonify({"error": "Valid email required"}), 400
        if minutes <= 0:
            return jsonify({"error": "Minutes must be positive"}), 400

        try:
            session = create_pay_per_use_session(
                minutes,
                email,
                success_url=f"{request.host_url.rstrip('/')}/success",
                cancel_url=f"{request.host_url.rstrip('/')}/pricing.html",
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except stripe.error.StripeError as exc:  # type: ignore[attr-defined]
            return jsonify({"error": str(exc)}), 400
        return jsonify({"checkout_url": session.url})

    @app.route("/api/pricing", methods=["GET"])
    def get_pricing():
        return jsonify({"plans": PLANS, "pay_per_use_rate": PAY_PER_USE_RATE_EUR_PER_MINUTE})

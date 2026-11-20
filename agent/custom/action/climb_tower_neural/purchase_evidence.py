"""Normalize one purchase debit from linked, observed wallet records.

Live capture and exports pass the same chronological wallet observations here.
The first complete wallet for the purchase operation precedes nested reward
refreshes; the pipeline's last wallet is a separate observation.
"""


def purchase_wallet_evidence(operation_id, price, wallets, *, before_wallet, pipeline_wallet):
    def amount(wallet):
        coin = wallet.get("coin") if wallet and wallet.get("coin_complete") and wallet.get("record_id") else None
        return coin if coin is not None and coin >= 0 else None

    after_wallet = next((wallet for wallet in wallets
                         if wallet.get("operation_id") == operation_id and amount(wallet) is not None), None)
    if (before_wallet and pipeline_wallet
            and before_wallet.get("record_id") == pipeline_wallet.get("record_id")):
        pipeline_wallet = None
    before, after = amount(before_wallet), amount(after_wallet)
    debit = before - after if before is not None and after is not None else None
    confirmed = debit == price if debit is not None else None
    return dict(coins_before=before, coins_after=after,
                purchase_debit=debit, debit_confirmed=confirmed,
                operation_confirmed=confirmed is True,
                confirmation_basis="linked_purchase_wallet_difference",
                confirmation_status=("wallet_missing" if confirmed is None else
                                     "confirmed" if confirmed else "amount_mismatch"),
                wallet_before_observation_id=(before_wallet or {}).get("record_id"),
                wallet_before_recognition_record_id=(before_wallet or {}).get("recognition_record_id"),
                wallet_observation_id=(after_wallet or {}).get("record_id"),
                wallet_recognition_record_id=(after_wallet or {}).get("recognition_record_id"),
                pipeline_coins_after=amount(pipeline_wallet),
                pipeline_wallet_observation_id=(pipeline_wallet or {}).get("record_id"),
                pipeline_wallet_recognition_record_id=(pipeline_wallet or {}).get("recognition_record_id"))

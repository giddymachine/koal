import argparse
import json
import random
import time
from math import ceil

import numpy as np
import pandas as pd
import requests
from jito_searcher_client import get_searcher_client
from jito_searcher_client.convert import versioned_tx_to_protobuf_packet
from jito_searcher_client.generated.bundle_pb2 import Bundle
from jito_searcher_client.generated.searcher_pb2 import (
    GetTipAccountsRequest,
    SendBundleRequest,
)
from solana.rpc.api import Client
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

LANDED = {"confirmed", "finalized"}


def parse_args():
    p = argparse.ArgumentParser(description="land rate vs tip size for self-transfer bundles")
    p.add_argument("--keypair", required=True, help="path to solana keypair json")
    p.add_argument("--rpc", required=True, help="rpc url")
    p.add_argument("--block-engine", default="amsterdam.mainnet.block-engine.jito.wtf")
    p.add_argument("--tips", default="1000,5000,10000,50000,120000", help="comma separated lamports")
    p.add_argument("--reps", type=int, default=20, help="bundles per tip level")
    p.add_argument("--gap", type=float, default=1.5, help="seconds between sends")
    p.add_argument("--settle", type=float, default=40.0, help="seconds to wait for results after last send")
    p.add_argument("--out", default="tip_sweep.csv")
    return p.parse_args()


def load_keypair(path):
    with open(path) as f:
        raw = json.load(f)
    return Keypair.from_bytes(bytes(raw))


def rpc(url, method, params):
    body = {"jsonrpc": "2.0", "id": int(time.time() * 1000), "method": method, "params": params}
    r = requests.post(url, json=body, timeout=10)
    return r.json()["result"]


def poll_statuses(host, ids, outcomes):
    url = f"https://{host}/api/v1/getBundleStatuses"
    for chunk in np.array_split(np.array(ids), ceil(len(ids) / 5)):
        try:
            result = rpc(url, "getBundleStatuses", [chunk.tolist()])
        except Exception as e:
            print(f"status poll failed: {e}")
            continue
        status = pd.DataFrame([v for v in result["value"] if v is not None])
        for entry in status.to_dict("records"):
            if entry.get("err") not in (None, {"Ok": None}):
                outcomes[entry["bundle_id"]] = "failed"
            else:
                outcomes[entry["bundle_id"]] = entry["confirmation_status"]


def build_tx(kp, rpc, tip_account, paid):
    bh = rpc.get_latest_blockhash().value.blockhash
    ixs = [
        transfer(TransferParams(from_pubkey=kp.pubkey(), to_pubkey=tip_account, lamports=paid)),
    ]
    msg = Message.new_with_blockhash(ixs, kp.pubkey(), bh)
    return VersionedTransaction(msg, [kp])


def main():
    args = parse_args()
    tips = [int(x) for x in args.tips.split(",")]
    kp = load_keypair(args.keypair)
    rpc = Client(args.rpc)
    client = get_searcher_client(args.block_engine, kp)

    tip_accounts = [Pubkey.from_string(a) for a in client.GetTipAccounts(GetTipAccountsRequest()).accounts]
    print(f"{len(tip_accounts)} tip accounts, wallet {kp.pubkey()}")

    outcomes = {}

    schedule = [tip for tip in tips for _ in range(args.reps)]
    random.shuffle(schedule)

    sent = []
    for i, tip in enumerate(schedule):
        paid = tip + random.randint(0, 50)
        tx = build_tx(kp, rpc, random.choice(tip_accounts), paid)
        bundle = Bundle(header=None, packets=[versioned_tx_to_protobuf_packet(tx)])
        try:
            uuid = client.SendBundle(SendBundleRequest(bundle=bundle)).uuid
        except Exception as e:
            print(f"send {i} failed at tip {tip}: {e}")
            sent.append({"bundle_id": f"send_error_{i}", "tip": tip, "paid": paid})
            continue
        sent.append({"bundle_id": uuid, "tip": tip, "paid": paid})
        print(f"[{i + 1}/{len(schedule)}] tip={tip} paid={paid} uuid={uuid}")
        time.sleep(args.gap)

    print(f"waiting {args.settle:.0f}s for results")
    deadline = time.time() + args.settle
    live = [s["bundle_id"] for s in sent if not s["bundle_id"].startswith("send_error")]
    while time.time() < deadline:
        todo = [b for b in live if outcomes.get(b) not in LANDED | {"failed"}]
        if not todo:
            break
        poll_statuses(args.block_engine, todo, outcomes)
        time.sleep(2)

    df = pd.DataFrame(sent)
    df["outcome"] = df["bundle_id"].map(outcomes).fillna("not_found")
    df.loc[df["bundle_id"].str.startswith("send_error"), "outcome"] = "send_error"
    df["landed"] = df["outcome"].isin(LANDED)
    df[["bundle_id", "tip", "paid", "outcome"]].to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")

    summary = df.groupby("tip").agg(sent=("landed", "size"), landed=("landed", "sum"), land_rate=("landed", "mean"))
    summary["land_rate"] = (summary["land_rate"] * 100).round(1)
    print(summary)
    print(f"overall land rate: {df['landed'].mean():.1%}")
    print(df.groupby(["tip", "outcome"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()

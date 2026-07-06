"""Seed-pair lifecycle tests (Rainbet-style reuse + on-demand rotation).

Pure/DB-free: exercises api.game.seedpair state transitions and confirms the
RNG stays deterministic across a reused pair with a climbing nonce.
"""

from api.game import dice, seedpair
from api.game.seed import rng_float, server_hash


def test_new_pair_shape_and_commitment():
    p = seedpair.new_pair()
    # Hashes commit to their seeds.
    assert p["server_hash"] == server_hash(p["server_seed"])
    assert p["next_server_hash"] == server_hash(p["next_server_seed"])
    assert p["nonce"] == 0
    assert p["client_seed"]


def test_new_pair_accepts_custom_client_seed():
    p = seedpair.new_pair("my-custom-seed")
    assert p["client_seed"] == "my-custom-seed"


def test_public_view_never_leaks_active_server_seed():
    p = seedpair.new_pair()
    view = seedpair.public_view(p)
    assert "server_seed" not in view
    assert "next_server_seed" not in view
    assert view["server_hash"] == p["server_hash"]
    assert view["next_server_hash"] == p["next_server_hash"]
    assert view["client_seed"] == p["client_seed"]
    assert view["nonce"] == 0


def test_pair_reuse_is_deterministic_across_climbing_nonce():
    """Same pair, advancing nonce → each bet's outcome is fixed and recomputable."""
    p = seedpair.new_pair("client-abc")
    ss, cs = p["server_seed"], p["client_seed"]
    outcomes = []
    for nonce in range(5):
        r = dice.settle(ss, cs, nonce, target=50)
        # Recomputing the same (seed, seed, nonce) reproduces the outcome exactly.
        assert dice.settle(ss, cs, nonce, target=50) == r
        outcomes.append(r["roll"])
    # Seeds are unchanged by "placing bets" (we only advance the nonce).
    assert p["server_seed"] == ss and p["client_seed"] == cs
    # Advancing the nonce changes the draw (not a stuck sequence).
    assert len(set(outcomes)) > 1


def test_rotation_reveals_old_promotes_next_and_resets_nonce():
    p = seedpair.new_pair("client-1")
    old_server_seed = p["server_seed"]
    committed_next_hash = p["next_server_hash"]

    new, revealed = seedpair.rotate(p, "client-2")

    # The retired server seed is revealed and matches the hash shown while active.
    assert revealed == old_server_seed
    assert server_hash(revealed) == p["server_hash"]
    # The pre-committed next seed becomes active; its hash was shown beforehand.
    assert new["server_hash"] == committed_next_hash
    assert server_hash(new["server_seed"]) == committed_next_hash
    # A fresh next seed is committed, the client seed applied, nonce reset.
    assert new["next_server_hash"] == server_hash(new["next_server_seed"])
    assert new["next_server_seed"] != p["next_server_seed"]
    assert new["client_seed"] == "client-2"
    assert new["nonce"] == 0


def test_rotation_without_client_seed_keeps_current():
    p = seedpair.new_pair("keep-me")
    new, _ = seedpair.rotate(p)
    assert new["client_seed"] == "keep-me"


def test_outcome_verifiable_after_rotation():
    """A player can reproduce a past bet once the server seed is revealed."""
    p = seedpair.new_pair("client-x")
    ss, cs = p["server_seed"], p["client_seed"]
    played = [dice.settle(ss, cs, n, target=50)["roll"] for n in range(3)]

    _, revealed = seedpair.rotate(p)
    # With the revealed seed + known client seed + nonces, every bet recomputes.
    recomputed = [dice.settle(revealed, cs, n, target=50)["roll"] for n in range(3)]
    assert recomputed == played
    # And the revealed seed matches the hash that was public during play.
    assert server_hash(revealed) == p["server_hash"]

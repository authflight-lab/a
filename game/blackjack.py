"""Blackjack — deal + hit/stand/double, single hand, no splits (spec: Task #4).

Standard rules, fixed (non-EPS-scaled) payouts, exactly like a real table:

    natural blackjack (2-card 21) beats a non-blackjack dealer  -> pays 3:2 (2.5x)
    both player and dealer have a natural                       -> push  (1.0x)
    player total > dealer total (dealer didn't bust)             -> pays 1:1 (2.0x)
    player busts (> 21) at any point                             -> loses (0x)
    dealer busts (player didn't)                                 -> pays 1:1 (2.0x)
    push (equal totals, no bust)                                 -> push  (1.0x)
    player total < dealer total                                  -> loses (0x)

Deck model: an "infinite deck" (draw-with-replacement) of ranks 1..13, each
equally likely — 4 of the 13 ranks (10, J, Q, K) are worth 10, matching a real
deck's ten-frequency (4/13) without the complexity of finite-deck depletion
across many rounds sharing one seed pair. This is the same simplifying choice
HighLow already makes for its rank draws (see highlow.draw_card).

Deal order (fixed rng cursor slots, dealt like a real table — player, dealer,
player, dealer): 0=player first card, 1=dealer up card, 2=player second card,
3=dealer hole card. Hits/dealer-draws consume cursor 4, 5, 6, ... in order.

Dealer plays fixed strategy: hit while total < 17, stand on any 17+ (S17 —
"dealer stands on 17" per the task spec, including a soft 17).
"""

RANKS = 13


def draw_rank(u: float) -> int:
    """Map an rng_float draw to a card rank in 1..13 (1=Ace, 10/11/12/13=ten-value)."""
    return int(u * RANKS) + 1


def card_value(rank: int) -> int:
    """Blackjack value of a rank, treating Ace as 11 (hand_total reduces it)."""
    if rank == 1:
        return 11
    return 10 if rank >= 10 else rank


def hand_total(cards: list[int]) -> tuple[int, bool]:
    """Best total for a hand, reducing Aces from 11 to 1 as needed.

    Returns ``(total, soft)`` where ``soft`` is True iff at least one Ace is
    still being counted as 11 (i.e. the hand isn't "hard")."""
    total = sum(card_value(c) for c in cards)
    aces_as_11 = cards.count(1)
    while total > 21 and aces_as_11 > 0:
        total -= 10
        aces_as_11 -= 1
    return total, aces_as_11 > 0


def is_bust(total: int) -> bool:
    return total > 21


def is_blackjack(cards: list[int]) -> bool:
    """A natural: exactly 2 cards totaling 21."""
    return len(cards) == 2 and hand_total(cards)[0] == 21


def dealer_should_hit(cards: list[int]) -> bool:
    """S17: hit on any total < 17, stand on 17+ (soft or hard)."""
    total, _ = hand_total(cards)
    return total < 17


def draw_card(draw_float, cursor: int) -> int:
    """``draw_float(i)`` maps an rng index to a uniform float in [0, 1)."""
    return draw_rank(draw_float(cursor))


def deal_initial(draw_float) -> dict:
    """Deal the opening 4 cards in real-table order and return the round state.

    ``next_cursor`` is where the first hit/dealer-draw will read from.
    """
    p1 = draw_card(draw_float, 0)
    d1 = draw_card(draw_float, 1)
    p2 = draw_card(draw_float, 2)
    d2 = draw_card(draw_float, 3)
    return {
        "player": [p1, p2],
        "dealer": [d1, d2],
        "next_cursor": 4,
        "doubled": False,
        "player_done": False,
    }


def play_dealer(draw_float, dealer: list[int], next_cursor: int) -> tuple[list[int], int]:
    """Draw for the dealer until they stand (or bust), returning (hand, cursor)."""
    dealer = list(dealer)
    cursor = next_cursor
    while dealer_should_hit(dealer):
        dealer.append(draw_card(draw_float, cursor))
        cursor += 1
    return dealer, cursor


def outcome_multiplier(player: list[int], dealer: list[int]) -> float:
    """Resolve a finished hand (player stood/busted, dealer played out).

    Naturals are compared as part of the same total-comparison logic: a 21
    from exactly 2 cards still just compares as 21 here — the 3:2 natural
    bonus is applied by the caller (main.py) ONLY when the round ends at the
    initial deal via a player natural, per real-table rules (a natural dealt
    after a hit/double never happens since it requires exactly 2 cards).
    """
    p_total, _ = hand_total(player)
    if is_bust(p_total):
        return 0.0
    d_total, _ = hand_total(dealer)
    if is_bust(d_total):
        return 2.0
    if p_total > d_total:
        return 2.0
    if p_total < d_total:
        return 0.0
    return 1.0


def natural_outcome(player: list[int], dealer: list[int]) -> float | None:
    """If the player was dealt a natural, resolve it immediately (real-table
    rule: a player blackjack is settled right away, not offered hit/stand).

    Returns the multiplier (2.5 = natural win, 1.0 = push vs dealer natural),
    or ``None`` if the player does NOT have a natural (round continues)."""
    if not is_blackjack(player):
        return None
    if is_blackjack(dealer):
        return 1.0
    return 2.5


def can_split(cards: list[int]) -> bool:
    """Split is offered only on the opening two IDENTICAL ranks (8+8 yes; K+Q
    no) — the user-chosen rule. Ten-value ranks (10/J/Q/K) are distinct ranks,
    so a K+Q or 10+J is NOT splittable, matching real identical-rank tables."""
    return len(cards) == 2 and cards[0] == cards[1]


def is_ace_pair(cards: list[int]) -> bool:
    """A pair of Aces gets special split handling: one card each, then stand."""
    return len(cards) == 2 and cards[0] == 1 and cards[1] == 1


def rtp_distribution(n_samples: int = 200_000) -> list[tuple[float, float]]:
    """Monte-Carlo outcome distribution under a simple always-stand-on-17-vs-
    hit-below-17 player strategy (mirrors the dealer's own fixed rule), used
    only as an RTP *sanity* check (spec note: blackjack pays REAL fixed odds —
    2x/2.5x/1x/0x — it is not EPS-scaled like the other games, so there is no
    single closed-form target to assert exact equality against; the test just
    asserts the simulated RTP lands in blackjack's well-known real-world band).

    Uses Python's own PRNG (not the seeded HMAC one) — fine for a statistical
    sanity check, not a fairness proof (that's covered by seed.py + the /me
    verify flow used by every other game already)."""
    import random

    dist: dict[float, int] = {}
    rng = random.Random(1234567)
    for _ in range(n_samples):
        deck_pos = [0]

        def draw_float(_cursor: int) -> float:
            deck_pos[0] += 1
            return rng.random()

        state = deal_initial(draw_float)
        player, dealer = state["player"], state["dealer"]
        nat = natural_outcome(player, dealer)
        cursor = state["next_cursor"]
        if nat is None:
            # Simple mimic-the-dealer player strategy: hit while < 17.
            while hand_total(player)[0] < 17:
                player.append(draw_card(draw_float, cursor))
                cursor += 1
            if is_bust(hand_total(player)[0]):
                mult = 0.0
            else:
                dealer, cursor = play_dealer(draw_float, dealer, cursor)
                mult = outcome_multiplier(player, dealer)
        else:
            mult = nat
        dist[mult] = dist.get(mult, 0) + 1
    total = float(n_samples)
    return [(count / total, mult) for mult, count in dist.items()]

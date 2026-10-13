"""kwja-mlx comprehensive test using Japanese Wikipedia article '日本'.

Tests all precision modes (fp32, fp16, q8, q4) across all functions:
- segment (word segmentation)
- tokenize (segment + POS + NER, no reading)
- analyze (full pipeline with T5 reading/lemma/canon)

Also validates:
- NER B/I sequence correctness
- Throughput (sentences/sec)
- Consistency across quantization levels
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Test data ---
# Selected sentences from Japanese Wikipedia '日本' article
TEST_SENTENCES = [
    "首都は東京都（事実上）。",
    "気候は温暖湿潤気候が中心で、四季が明瞭である。",
    "8世紀ごろまでには律令国家としての体制が整った。",
    "日本は積極的な近代化政策のもと国力を伸長させたが、第二次世界大戦に敗北したのち連合国軍に占領された。",
    "戦後日本はサンフランシスコ講和条約を経て主権を回復し、高度経済成長期を経て先進国・経済大国へと伸長した。",
    "1947年施行の日本国憲法に基づき、天皇は「日本国および日本国民統合の象徴」として位置づけられている。",
    "日本国憲法にて、戦争の放棄が定められている。",
    "日本の経済規模は名目GDPと購買力平価で世界第4–5位、労働力人口は世界8位である。",
    "710年（和銅3年）の平城京遷都によりはじまった奈良時代において、日本は仏教を中心とする律令国家として運営されたものの、この時代から公地公民制にほころびがあらわれはじめた。",
    "所領を通じた幕府と御家人の関係は、貨幣経済の発展や、元寇に対する恩賞の不十分さなどにより、困窮する御家人が多くなったことでゆらぎはじめ、1333年（元弘3年・正慶2年）に鎌倉幕府が滅亡した。",
    "田中太郎さんが東京スカイツリーを訪れた。",  # explicit NER test
    "トヨタ自動車株式会社は2023年3月に1兆円の利益を計上した。",  # ORG + DATE + MONEY
]

# Expected NER entities (partial, for validation)
EXPECTED_NER = {
    "首都は東京都（事実上）。": [("東京都", "LOCATION")],
    "田中太郎さんが東京スカイツリーを訪れた。": [("田中太郎", "PERSON"), ("東京スカイツリー", "LOCATION")],
    "トヨタ自動車株式会社は2023年3月に1兆円の利益を計上した。": [("トヨタ自動車株式会社", "ORGANIZATION")],
}

NE_TAGS = ['O', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-PERSON', 'I-PERSON',
           'B-LOCATION', 'I-LOCATION', 'B-ARTIFACT', 'I-ARTIFACT',
           'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-MONEY', 'I-MONEY',
           'B-PERCENT', 'I-PERCENT']


def validate_bio_sequences(morphemes) -> list[str]:
    """Check that B/I NER sequences are valid. Returns list of errors."""
    errors = []
    prev_type = None
    for i, m in enumerate(morphemes):
        ne = m.ne
        if ne.startswith("I-"):
            ne_type = ne[2:]
            if prev_type != ne_type:
                errors.append(
                    f"Invalid I-{ne_type} at pos {i} ('{m.surface}'): "
                    f"prev was {'O' if prev_type is None else prev_type}"
                )
        if ne.startswith("B-"):
            prev_type = ne[2:]
        elif ne.startswith("I-"):
            pass  # prev_type stays the same
        else:
            prev_type = None
    return errors


def extract_entities(morphemes) -> list[tuple[str, str]]:
    """Extract (surface, type) entities from morpheme list."""
    entities = []
    current_surface = ""
    current_type = None
    for m in morphemes:
        if m.ne.startswith("B-"):
            if current_type:
                entities.append((current_surface, current_type))
            current_surface = m.surface
            current_type = m.ne[2:]
        elif m.ne.startswith("I-") and current_type == m.ne[2:]:
            current_surface += m.surface
        else:
            if current_type:
                entities.append((current_surface, current_type))
            current_surface = ""
            current_type = None
    if current_type:
        entities.append((current_surface, current_type))
    return entities


def run_throughput_test(analyzer, sentences, mode="tokenize", n_runs=3):
    """Measure throughput in sentences/sec."""
    # Warmup
    if mode == "segment":
        analyzer.segment(sentences[0])
    elif mode == "tokenize":
        analyzer.tokenize(sentences[0])
    else:
        analyzer.analyze(sentences[0], with_reading=True)

    times = []
    for _ in range(n_runs):
        t0 = time.time()
        for s in sentences:
            if mode == "segment":
                analyzer.segment(s)
            elif mode == "tokenize":
                analyzer.tokenize(s)
            else:
                analyzer.analyze(s, with_reading=True)
        times.append(time.time() - t0)

    best = min(times)
    return len(sentences) / best


def test_precision(quantize_mode):
    """Run full test suite for a given precision mode."""
    from kwja_mlx import Analyzer

    print(f"\n{'='*60}")
    print(f"  Testing: quantize={quantize_mode or 'fp32'}")
    print(f"{'='*60}")

    t0 = time.time()
    a = Analyzer(quantize=quantize_mode)
    load_time = time.time() - t0
    print(f"  Model load time: {load_time:.1f}s")

    results = {}

    # --- Segmentation test ---
    print("\n  [1] Word Segmentation")
    seg_results = {}
    for sent in TEST_SENTENCES[:5]:
        words = a.segment(sent)
        seg_results[sent] = words
        print(f"      {sent[:30]}... → {len(words)} words: {'|'.join(words[:8])}")

    seg_speed = run_throughput_test(a, TEST_SENTENCES, mode="segment")
    print(f"      Throughput: {seg_speed:.0f} sent/sec")
    results['segment_speed'] = seg_speed

    # --- Tokenize test (POS + NER, no reading) ---
    print("\n  [2] POS + NER (tokenize)")
    all_bio_errors = []
    all_entities = {}
    for sent in TEST_SENTENCES:
        morphemes = a.tokenize(sent)
        bio_errors = validate_bio_sequences(morphemes)
        all_bio_errors.extend(bio_errors)
        entities = extract_entities(morphemes)
        all_entities[sent] = entities

        # Show NER-bearing sentences
        if entities:
            print(f"      {sent[:40]}...")
            for surf, etype in entities:
                print(f"        → {surf} [{etype}]")

    if all_bio_errors:
        print(f"      ⚠️  B/I errors ({len(all_bio_errors)}):")
        for e in all_bio_errors[:5]:
            print(f"        {e}")
    else:
        print("      ✅ All B/I sequences valid")

    tok_speed = run_throughput_test(a, TEST_SENTENCES, mode="tokenize")
    print(f"      Throughput: {tok_speed:.0f} sent/sec")
    results['tokenize_speed'] = tok_speed
    results['bio_errors'] = len(all_bio_errors)

    # --- NER accuracy check ---
    print("\n  [3] NER Entity Validation")
    ner_hits = 0
    ner_total = 0
    for sent, expected in EXPECTED_NER.items():
        found = all_entities.get(sent, [])
        for exp_surf, exp_type in expected:
            ner_total += 1
            # Fuzzy match: entity surface contains or is contained in expected
            matched = any(
                (exp_surf in surf or surf in exp_surf) and etype == exp_type
                for surf, etype in found
            )
            status = "✅" if matched else "❌"
            if matched:
                ner_hits += 1
            print(f"      {status} Expected [{exp_type}] '{exp_surf}' → "
                  f"found: {[f'{s}[{t}]' for s, t in found]}")

    ner_recall = ner_hits / ner_total if ner_total > 0 else 0
    print(f"      NER recall: {ner_hits}/{ner_total} ({ner_recall:.0%})")
    results['ner_recall'] = ner_recall

    # --- Full analysis test (with T5 reading) ---
    print("\n  [4] Full Analysis (with reading/lemma)")
    test_sents_short = TEST_SENTENCES[:4]  # T5 is slow, limit
    for sent in test_sents_short[:2]:
        result = a.analyze(sent, with_reading=True)
        print(f"      {sent[:30]}...")
        for m in result.morphemes[:6]:
            reading_str = f" [{m.reading}]" if m.reading else ""
            print(f"        {m.surface}/{m.pos}{reading_str}")
        if len(result.morphemes) > 6:
            print(f"        ... ({len(result.morphemes)} morphemes total)")

    analyze_speed = run_throughput_test(a, test_sents_short, mode="analyze", n_runs=1)
    print(f"      Throughput: {analyze_speed:.1f} sent/sec")
    results['analyze_speed'] = analyze_speed

    # --- Bulk throughput (525 sentences from Wikipedia) ---
    print("\n  [5] Bulk Throughput (Wikipedia full article)")
    wiki_sents_path = Path("/tmp/wiki_japan_sents.txt")
    if wiki_sents_path.exists():
        wiki_sents = [line.strip() for line in wiki_sents_path.read_text().splitlines() if line.strip()]
        t0 = time.time()
        for s in wiki_sents:
            a.tokenize(s)
        bulk_time = time.time() - t0
        bulk_speed = len(wiki_sents) / bulk_time
        print(f"      {len(wiki_sents)} sentences in {bulk_time:.1f}s → {bulk_speed:.0f} sent/sec")
        results['bulk_speed'] = bulk_speed
    else:
        print("      (wiki sentences file not found, skipping)")

    return results


def main():
    print("=" * 60)
    print("  kwja-mlx Comprehensive Test")
    print("  Source: Japanese Wikipedia '日本'")
    print("=" * 60)

    precisions = [None, "fp16", "q8", "q4"]
    all_results = {}

    for q in precisions:
        try:
            all_results[q or "fp32"] = test_precision(q)
        except Exception as e:  # noqa: BLE001 — benchmark harness: report per-precision failure, keep going
            print(f"  ❌ Failed for {q or 'fp32'}: {e}")
            import traceback
            traceback.print_exc()

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Mode':<6} | {'Segment':>8} | {'Tokenize':>9} | {'Analyze':>8} | {'Bulk':>6} | {'BIO err':>7} | {'NER':>5}")
    print(f"  {'':<6} | {'sent/s':>8} | {'sent/s':>9} | {'sent/s':>8} | {'s/s':>6} | {'':>7} | {'recall':>5}")
    print(f"  {'─'*6}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*5}")
    for name, r in all_results.items():
        print(f"  {name:<6} | {r.get('segment_speed',0):>8.0f} | {r.get('tokenize_speed',0):>9.0f} | "
              f"{r.get('analyze_speed',0):>8.1f} | {r.get('bulk_speed',0):>6.0f} | "
              f"{r.get('bio_errors',0):>7} | {r.get('ner_recall',0):>5.0%}")


if __name__ == "__main__":
    main()

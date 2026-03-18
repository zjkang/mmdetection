"""Generate Minimum Distinguishing Attributes (MDA) for confusable class pairs.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python tools/mda/generate_mda.py \
        --pairs data/mda/confusable_pairs.json \
        --output data/mda/mda_attributes.json \
        --batch-size 20
"""
import argparse
import json
import os
import time

import anthropic

PROMPT_TEMPLATE = """You are a visual attribute expert. For each pair of visually confusable object classes below, provide the **minimum distinguishing visual attribute** — the single most reliable visual cue that distinguishes class A from class B in a photograph.

Rules:
- Focus on VISUAL attributes only (shape, color, texture, size, parts, material)
- Be concise: each attribute should be 5-15 words
- The attribute must be discriminative: true for one class but not the other

For each pair, output JSON with this exact format:
{{"pair": "class_a|class_b", "attr_a": "...", "attr_b": "..."}}

Where attr_a distinguishes class_a FROM class_b, and attr_b distinguishes class_b FROM class_a.

Pairs:
{pairs_text}

Output a JSON array of all pairs. Output ONLY the JSON array, no other text."""


def build_pairs_text(pairs_batch):
    lines = []
    for cls_a, cls_b in pairs_batch:
        lines.append(f"- {cls_a} vs {cls_b}")
    return "\n".join(lines)


def call_claude(client, pairs_batch, model="claude-haiku-4-5-20251001"):
    pairs_text = build_pairs_text(pairs_batch)
    prompt = PROMPT_TEMPLATE.format(pairs_text=pairs_text)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def parse_response(text):
    # Find JSON array in response
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code block
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs", default="data/mda/confusable_pairs.json")
    parser.add_argument(
        "--output", default="data/mda/mda_attributes.json")
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Number of pairs per API call")
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Claude model to use")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)

    # Load confusable pairs
    with open(args.pairs) as f:
        confusable = json.load(f)

    # Flatten to list of (class_a, class_b) pairs
    all_pairs = []
    for cls_a, negatives in confusable.items():
        for cls_b in negatives:
            all_pairs.append((cls_a, cls_b))

    print(f"Total pairs: {len(all_pairs)}")

    # Load existing results if resuming
    results = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} pairs already done")

    # Filter out already-done pairs
    remaining = [(a, b) for a, b in all_pairs
                 if f"{a}|{b}" not in results]
    print(f"Remaining pairs: {len(remaining)}")

    # Process in batches
    for i in range(0, len(remaining), args.batch_size):
        batch = remaining[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
        print(f"Batch {batch_num}/{total_batches} "
              f"({len(batch)} pairs)...", end=" ", flush=True)

        try:
            response_text = call_claude(client, batch, model=args.model)
            parsed = parse_response(response_text)

            for item in parsed:
                key = item["pair"]
                results[key] = {
                    "attr_a": item["attr_a"],
                    "attr_b": item["attr_b"],
                }

            print(f"OK ({len(parsed)} parsed)")

        except Exception as e:
            print(f"ERROR: {e}")
            # Save progress and continue
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            time.sleep(5)
            continue

        # Save progress after each batch
        if batch_num % 5 == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

        # Rate limiting
        time.sleep(0.5)

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! {len(results)} pairs saved to {args.output}")


if __name__ == "__main__":
    main()

import json

from django.core.management.base import BaseCommand

from core import rag


class Command(BaseCommand):
    help = "Evaluate retrieval on eval/rag_questions.json: hit@k for answerable questions, refusal rate for out-of-scope ones."

    def add_arguments(self, parser):
        parser.add_argument("--k", type=int, default=3)
        parser.add_argument("--threshold", type=float, default=None, help="override RAG_MIN_SCORE")
        parser.add_argument("--sweep", action="store_true", help="try several thresholds")

    def handle(self, *args, **opts):
        if opts["sweep"]:
            self.stdout.write("threshold  hit@k  refusal")
            for t in (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
                r = rag.evaluate(k=opts["k"], min_score=t)
                hit = r["hit_at_%d" % opts["k"]]
                self.stdout.write(f"  {t:.2f}     {hit:.3f}  {r['refusal_rate']:.3f}")
            return
        r = rag.evaluate(k=opts["k"], min_score=opts["threshold"])
        self.stdout.write(json.dumps(r, indent=2, ensure_ascii=False))

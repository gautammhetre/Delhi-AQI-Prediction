from django.core.management.base import BaseCommand

from core import ml
from predictor import services


class Command(BaseCommand):
    help = "Retrain the PM2.5 models on data/delhi_aqi.csv, evaluate them on the most recent 20% of the timeline, and save the chosen one."

    def handle(self, *args, **options):
        self.stdout.write("Training and evaluating (time-based split)...")
        artifact, metrics = ml.train_and_evaluate(log=self.stdout.write)
        ml.save(artifact, metrics)
        services.reset_predictor()
        best = metrics["models"][metrics["best_model"]]
        self.stdout.write(self.style.SUCCESS(
            f"Saved {metrics['best_model']}: MAE {best['mae']} µg/m³, category accuracy {best['category_accuracy']:.1%} "
            f"on {metrics['test']['rows']} unseen hours ({metrics['test']['from'][:10]} to {metrics['test']['to'][:10]})."))

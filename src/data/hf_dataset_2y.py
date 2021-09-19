# Loading script for the HF dataset.
import datasets
import csv

logger = datasets.logging.get_logger(__name__)

_CITATION = """ """

_DESCRIPTION = """
               """

_HOMEPAGE = """None"""

_URL = "../../data/"
_TRAINING_FILE = "2019-02-11to2021-02-11_deduped.csv"
_TEST_FILE = "2021-02-12to2021-02-26_deduped.csv"


class HFConfig(datasets.BuilderConfig):
    """ Builder config for the HF dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for HF.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(HFConfig, self).__init__(**kwargs)


class HF(datasets.GeneratorBasedBuilder):
    """ HF dataset."""

    BUILDER_CONFIGS = [
        HFConfig(
            name="HF",
            version=datasets.Version("1.0.0"),
            description="HF dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.features.Value('float')
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            #"dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            #datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            reader = csv.reader(f, delimiter=',', quotechar='"')
            next(reader, None)
            for line in reader:
                yield guid, {
                    "id": str(guid),
                    "sentence": line[4],
                    "label": float(line[3])
                }
                guid += 1

from __future__ import absolute_import, division, print_function

import pickle
import logging
import datasets
import numpy as np


_CITATION = """None
"""

_DESCRIPTION = """\
Takes as input a pickled dict of pt2stream. The format should be:
    {'patient_id': (concept_cui, concept_count_for_patient, timestamp_of_first_occurrence_for_patient), ...}
"""

class PatientConceptStreamConfig(datasets.BuilderConfig):
    """ BuilderConfig for PatientConceptStream.

        Args:
            **kwargs: keyword arguments forwarded to super.
    """

    def __init__(self, **kwargs):
        super(PatientConceptStreamConfig, self).__init__(**kwargs)


class PatientConceptStream(datasets.GeneratorBasedBuilder):
    """PatientConceptStream: as input takes the patient to stream of concepts.

    TODO: Move the preparations scripts out of notebooks
    """

    BUILDER_CONFIGS = [
        PatientConceptStreamConfig(
            name="pickle",
            version=datasets.Version("1.0.0", ""),
            description="Pickled output from Temporal dataset preparation scripts",
        ),
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "patient_id": datasets.Value("string"),
                    "stream": [
                        {
                            'token': datasets.Value('string'),
                            'cui': datasets.Value('string'),
                            'cnt': datasets.Value('int32'),
                            'time': datasets.Value('int64'),
                            'token_type': datasets.Value('string'),
                            'doc_id': datasets.Value('string'),
                            'cntx_left': datasets.Sequence(datasets.Value('string')),
                            'cntx_left_inds': datasets.Sequence(datasets.Value('int32')),
                            'cntx_right': datasets.Sequence(datasets.Value('string')),
                            'cntx_right_inds': datasets.Sequence(datasets.Value('int32')),
                            'presence': datasets.Value('string'),
                            'ent_tkn_id': datasets.Value('int32'),
                        }
                    ],
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepaths': self.config.data_files['train'],
                },
            ),
        ]

    def _generate_examples(self, filepaths):
        """Returns Pts one by one"""
        for filepath in filepaths:
            logging.info("generating examples from = %s", filepath)
            with open(filepath, 'rb') as f:
                pt2stream = pickle.load(f)
                for pt, stream in pt2stream.items():
                    out_stream = []
                    # If time == None there is no temporal info, and for this dataset it is required
                    stream = [data for data in stream if data[2] is not None]
                    # Sort the stream by time - ascending
                    stream.sort(key=lambda data: data[3])
                    for data in stream:
                        out_stream.append({
                                           'token': data[0], # Potentially modified cui
                                           'cui': data[1], # the CUI for this token, as tokens can be slightly modified versions
                                           'cnt': data[2],
                                           'time': int(data[3]), # We convert this into int for speed
                                           'token_type': data[4], # Call it token from now on as it does not have to be only CUIs
                                           'doc_id': data[5],
                                           'cntx_left': data[6],
                                           'cntx_left_inds': data[7],
                                           'cntx_right': data[8],
                                           'cntx_right_inds': data[9],
                                           'presence': data[10],
                                           'ent_tkn_id': data[11],
                                           })
                    pt = str(pt)
                    if out_stream: # Skip streams that have zero annotations
                        yield pt, {'patient_id': str(pt),
                                   'stream': out_stream}

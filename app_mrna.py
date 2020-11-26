"""Part I: CellQuantifier Sequence Control"""

control = [

'segm_cell',
'classify_cell',
'detect_miR146',
'detect_miR155',

]

"""Part II: CellQuantifier Parameter Settings"""

settings = {

  #IO
  'IO input_path': '/home/linhua/Desktop/output/',
  'IO output_path': '/home/linhua/Desktop/output/',
  'Processed By:': 'Hua Lin',
  'Pixel_size': 0.0295,
  'Frame_rate': 1,

  #FILE LABELS
  'Dapi mask file label': 'mask.tif',
  'Dapi marker file label': 'mask.txt',
  'Dapi channel file label': 'C3-',
  'INS channel file label': 'C4-',
  'miR146 channel file label': 'C1-',
  'miR155 channel file label': 'C2-',

  #FOCI DETECTION SETTINGS
  'Foci det blob_thres_rel': 0.7,
  'Foci det blob_min_sigma': 2,
  'Foci det blob_max_sigma': 20,
  'Foci det blob_num_sigma': 10,
  'Foci det pk_thres_rel': 0,

}

"""Part III: Run CellQuantifier"""
from cellquantifier.util.pipeline_mrna import *
pipeline_batch(settings, control)

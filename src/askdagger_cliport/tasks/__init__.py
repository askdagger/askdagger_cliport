# credit: https://github.com/cliport/cliport

"""Ravens tasks."""

from askdagger_cliport.tasks.packing_shapes import PackingShapesOriginal
from askdagger_cliport.tasks.packing_shapes import PackingSeenShapes
from askdagger_cliport.tasks.packing_shapes import PackingUnseenShapes
from askdagger_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsOriginalSeq
from askdagger_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsOriginalSeq
from askdagger_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsOriginalGroup
from askdagger_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsOriginalGroup
from askdagger_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from askdagger_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from askdagger_cliport.tasks.packing_google_objects import PackingSeenGoogleObjectsGroup
from askdagger_cliport.tasks.packing_google_objects import PackingUnseenGoogleObjectsGroup
from askdagger_cliport.tasks.put_block_in_bowl import PutBlockInBowlSeenColors
from askdagger_cliport.tasks.put_block_in_bowl import PutBlockInBowlUnseenColors
from askdagger_cliport.tasks.put_block_in_bowl import PutBlockInBowlFull

names = {
    # goal conditioned
    "packing-shapes-original": PackingShapesOriginal,
    "packing-shapes": PackingSeenShapes,
    "packing-seen-shapes": PackingSeenShapes,
    "packing-unseen-shapes": PackingUnseenShapes,
    "packing-seen-google-objects-original-seq": PackingSeenGoogleObjectsOriginalSeq,
    "packing-unseen-google-objects-original-seq": PackingUnseenGoogleObjectsOriginalSeq,
    "packing-seen-google-objects-original-group": PackingSeenGoogleObjectsOriginalGroup,
    "packing-unseen-google-objects-original-group": PackingUnseenGoogleObjectsOriginalGroup,
    "packing-seen-google-objects-seq": PackingSeenGoogleObjectsSeq,
    "packing-unseen-google-objects-seq": PackingUnseenGoogleObjectsSeq,
    "packing-seen-google-objects-group": PackingSeenGoogleObjectsGroup,
    "packing-unseen-google-objects-group": PackingUnseenGoogleObjectsGroup,
    "put-block-in-bowl-seen-colors": PutBlockInBowlSeenColors,
    "put-block-in-bowl-unseen-colors": PutBlockInBowlUnseenColors,
    "put-block-in-bowl-full": PutBlockInBowlFull,
}

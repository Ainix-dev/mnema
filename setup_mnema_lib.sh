#!/bin/bash
# setup_mnema_lib.sh
# Run from /home/nehtrm/
# Copies existing MNEMA components into the new library structure

SRC="/home/nehtrm/liquid_memory"
DST="/home/nehtrm/mnema-lib"

echo "Copying memory components..."
cp $SRC/memory/graph.py      $DST/mnema/memory/
cp $SRC/memory/extractor.py  $DST/mnema/memory/
cp $SRC/memory/composer.py   $DST/mnema/memory/
cp $SRC/memory/fade.py       $DST/mnema/memory/

echo "Copying cognition components..."
cp $SRC/memory/goals.py      $DST/mnema/cognition/
cp $SRC/memory/metacog.py    $DST/mnema/cognition/
cp $SRC/memory/asc.py        $DST/mnema/cognition/

echo "Copying system components..."
cp $SRC/memory/hardware.py   $DST/mnema/system/
cp $SRC/scheduler.py         $DST/mnema/system/scheduler.py

echo "Copying consolidation..."
cp $SRC/consolidation/trainer.py $DST/mnema/consolidation/
cp $SRC/consolidation/ewc.py     $DST/mnema/consolidation/

echo "Done. Now update imports in each file:"
echo "  'from config import cfg' → 'from ..config import MNEMAConfig'"
echo "  relative imports between subpackages"

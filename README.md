# Spot Whiteboard

CS 5190 Computer Vision (CV) group project at Cal Poly Pomona. A Boston Dynamics Spot robot identifies a whiteboard, flattens it to a straight-on view, and extracts what's written on it.

Initially, Spot will detect characters written in the English language.

- **Mini 1:** conventional CV (edges, contours, homography, lighting correction, thresholding)
- **Mini 2:** deep learning (board detection + handwriting OCR)
- **Final:** run both on Spot, compare, write the paper

## Pipeline

```
photo → 1. detect board → 2. flatten → 3. reduce glare → 4. find writing → (Mini 2: read text)
```

| Step | Starter method | Function in `whiteboard_starter.py` |
|---|---|---|
| 1. Detect board | Canny edges + largest 4-sided contour | `detect_board` |
| 2. Flatten | 4-point homography warp | `flatten` |
| 3. Reduce glare | Divide by estimated background lighting | `normalize_lighting` |
| 4. Find writing | Adaptive threshold + dilation into line boxes | `find_writing` |

## Week 1: prototype bake-off

1. **Take 4 whiteboard photos on campus.** Mix: straight, angled, far, glare, messy writing.
2. **Upload them to the shared Drive folder** using the naming format below.
3. **Set up and run the starter:**
   ```bash
   git clone <repo-url> && cd spot-whiteboard
   pip install -r requirements.txt
   # download ALL photos from the shared folder into data/phone_photos/
   python whiteboard_starter.py data/phone_photos results
   ```
4. **Pick ONE step to improve** (see `prototypes/README.md`).
5. **Fill in your row in `RESULTS.md`** before the next meeting.

## No photos yet? Use the synthetic set

```bash
python make_test_photos.py data/synthetic            # 16 photos + ground truth (~30 s)
python whiteboard_starter.py data/synthetic results
python evaluate.py whiteboard_starter.py             # scores every step with real numbers
python evaluate.py prototypes/<you>/<you>_pipeline.py  # compare yours
```

`evaluate.py` scores Step 1 by corner error and Steps 3+4 by ink F1 against the true writing mask, so each person can prove their step improved.

## Photo naming

```
name_##_tags.jpg
```

- `name`: your first name, lowercase
- `##`: 01–04
- `tags`: underscores between conditions, from this list:
  `straight`, `angle30`, `angle45`, `angle60`, `close`, `far`, `glare`, `dim`, `messy`, `clean`, `colored`, `partial`, `clutter`

Examples: `mazen_01_straight_clean.jpg`, `mazen_02_angle45_glare.jpg`, `mazen_03_far_messy.jpg`

## Repo layout

```
data/phone_photos/       put photos here (do NOT commit photos — they live in Drive)
prototypes/<name>/       your improved version of one step
.gitignore               commits photos
README.md               initial document to read on project progress
RESULTS.md               comparison table for the meeting
evaluate.py              scores a pipeline against ground truth
make_test_photos.py      synthetic photos + ground truth
requirements.txt           libraries to include (?)
week1_results.html                 script output (not committed)
whiteboard_starter.py    shared baseline everyone runs
```

## Rules

- Don't edit `whiteboard_starter.py` directly. Copy it into `prototypes/<name>/`.
- Don't commit photos or results images. They're in `.gitignore`.
- Work on a branch named after you, then open a pull request.

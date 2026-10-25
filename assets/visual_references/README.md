# Chispa-Pifia Visual Bible

## Canonical reference files
Put the two approved reference images in this folder using these exact filenames:

- `chispa_character_standing_poses.jpg` — character identity, proportions, outfit, front/side/back views, expressions and pose range.
- `chispa_environment_style_reference.jpg` — detailed locations, cinematic lighting, colour palette, camera language and overall rendering mood.

The image renderer loads both files and sends them with each image-generation request. The first image anchors the character; the second anchors environments and cinematography. The previously accepted frame is also supplied to help preserve continuity from beat to beat.

## Main character identity lock
- Large smooth pale round head; large oval black eyes; expressive bold black eyebrows; small simple mouth.
- Tiny neck, developed compact torso, articulated arms/hands/legs/feet: more developed than line-only stick figures, but unmistakably a round-headed stickman.
- Default outfit: dark charcoal/black hoodie with drawstrings, dark cargo trousers, black-and-white high-top sneakers.
- Dark backpack only when the story establishes that the character is carrying it.
- Preserve head-to-body ratio, eye spacing, limb thickness, silhouette, clothing and footwear across all scenes.
- Do not invent facial hair, realistic nose/skin detail, logos, random accessories or unexplained outfit changes.

## Environment and rendering
- Detailed, believable locations with depth, practical objects, reflections, shadows and atmospheric perspective.
- Cinematic stylized animation: soft dimensional shading and clean outlines on the character; detailed cinematic environments.
- Dark charcoal, black, cool blue and steel-grey base palette with restrained warm amber/orange light accents.
- Light direction, room layout, doors/windows, prop placement, time of day and object state must persist between adjacent shots unless the story changes them.

## Story-frame rules
- Output one full-frame vertical 9:16 image per beat, never a reference-sheet collage.
- Choose the camera angle for the beat: wide establishing, medium action, face close-up, over-the-shoulder, POV insert, high/low angle or detail shot.
- Pose and expression must communicate the exact narrated action and emotion.
- No text, captions, subtitles, labels, readable signs, logos, watermarks or typography in generated images.
- The story is the source of truth. Reference-sheet example scenes are style examples only and must not leak into unrelated stories.

## Important limitations
Reference images improve consistency but do not guarantee perfect image identity. The pipeline must be checked by rendering several adjacent scenes and visually reviewing them before trusting a full production run. If the canonical JPEGs are absent, the script falls back to the legacy embedded reference; that fallback is not the final approved setup.

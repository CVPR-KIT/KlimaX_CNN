"""Corrected forecast maps only, from native 500m fields. No truth input."""
from datetime import datetime, timedelta
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Same physical rain classes as the comparison maps; expressed in mm/h.
BINS = np.array([0, .6, 1.2, 3, 6, 12, 18, 30, 60, 120], dtype=float)
COLORS = np.array([(255,255,255), (185,228,240), (104,184,220), (42,128,194),
                   (65,173,97), (165,200,75), (243,201,67), (233,137,56),
                   (166,51,101), (87,35,116)], dtype=np.uint8)


def _font(size):
    path = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/'arial.ttf'
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def export_pngs(corrected_mm10, outroot, first_forecast):
    values = np.asarray(corrected_mm10)
    if values.shape != (18, 100, 100) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('PNG export requires 18 finite nonnegative 100x100 predictions')
    if not isinstance(first_forecast, datetime):
        raise TypeError('first_forecast must be datetime')
    folder = Path(outroot)/'png_corrected'
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError('Refusing to overwrite PNG results: ' + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    title_font, label_font, small_font = _font(25), _font(19), _font(16)
    for index, field in enumerate(values):
        stamp = first_forecast + timedelta(minutes=10*index)
        lead = 10*(index+1)
        # Convert once for display. Native row indices increase northward.
        hourly = field.astype(np.float64)*6.
        classes = np.searchsorted(BINS[1:], hourly, side='right')
        pixels = COLORS[classes]
        rain_map = Image.fromarray(np.flipud(pixels), mode='RGB').resize((600,600), Image.Resampling.NEAREST)
        image = Image.new('RGB', (940,830), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((70,22), 'Corrected rainfall forecast', fill='black', font=title_font)
        draw.text((70,60), f'{stamp:%Y-%m-%d %H:%M}   |   lead +{lead:03d} min', fill='black', font=label_font)
        draw.text((70,92), 'Native 500 m grid / 50 km area / fixed colour scale', fill='#555555', font=small_font)
        image.paste(rain_map, (70,140))
        draw.rectangle((70,140,670,740), outline='black', width=1)
        for km in (-25,-10,0,10,25):
            x = 70+int((km+25)*12)
            y = 140+int((25-km)*12)
            draw.line((x,740,x,747), fill='black', width=1)
            draw.text((x,751), str(km), fill='black', font=small_font, anchor='mt')
            draw.line((63,y,70,y), fill='black', width=1)
            draw.text((57,y), str(km), fill='black', font=small_font, anchor='rm')
        draw.text((370,782), 'x (km)', fill='black', font=label_font, anchor='mt')
        draw.text((8,111), 'y (km)', fill='black', font=small_font)
        draw.text((710,140), 'Rainfall (mm/h)', fill='black', font=label_font)
        for j in range(len(COLORS)):
            y = 187+j*45
            colour = tuple(int(c) for c in COLORS[j])
            draw.rectangle((715,y,747,y+25), fill=colour, outline='#777777')
            label = f'{BINS[j]:g} - {BINS[j+1]:g}' if j+1 < len(BINS) else f'>= {BINS[j]:g}'
            draw.text((760,y+2), label, fill='black', font=small_font)
        draw.text((730,670), 'N', fill='black', font=label_font)
        draw.line((738,729,738,700), fill='black', width=2)
        draw.polygon([(738,695),(733,706),(743,706)], fill='black')
        path = folder/f'pred_corrected_{stamp:%Y%m%d_%H%M}_lead{lead:03d}min.png'
        image.save(path, format='PNG')
    return folder

"""Progressive three-station plots from evaluation records, without NC access."""
from datetime import datetime, timedelta
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size):
    path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'arial.ttf'
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def export_station_graphs(evaluation, outroot):
    """Write 18 progressive plots with fixed time and per-station rain axes."""
    if evaluation.get('status') != 'available':
        return None
    records = evaluation.get('rows')
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError('Graphs require three evaluation station records')
    if [str(record.get('station_id')) for record in records] != ['410', '418', '889']:
        raise ValueError('Graph station order must be 410, 418, 889')
    series, timestamps, maxima = [], None, []
    for record in records:
        rows = record.get('rows')
        if not isinstance(rows, list) or len(rows) != 18:
            raise ValueError('Graphs require 18 rows per station')
        times = [datetime.fromisoformat(row['timestamp']) for row in rows]
        if any(times[i] != times[0] + timedelta(minutes=10*i) for i in range(18)):
            raise ValueError('Graph timestamps must be consecutive ten-minute intervals')
        if timestamps is not None and times != timestamps:
            raise ValueError('Station timestamps must match')
        timestamps = times
        points = []
        for row in rows:
            truth, predicted = float(row['truth_mm10']), float(row['prediction_mm10'])
            if not all(math.isfinite(v) and v >= 0 for v in (truth, predicted)):
                raise ValueError('Graph rainfall must be finite and nonnegative')
            points.append((truth, predicted))
        series.append(points)
        maxima.append(max(0.1, max(max(t*1.5, p) for t, p in points)*1.12))
    folder = Path(outroot) / 'graphs'
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError('Refusing to overwrite station graphs: ' + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    title_font, font, small = _font(25), _font(18), _font(15)
    left, right = 100, 1100
    for frame, stamp in enumerate(timestamps):
        image = Image.new('RGB', (1200, 1050), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((55, 20), 'Rainfall at station grid cells: truth and corrected forecast', fill='black', font=title_font)
        draw.text((55, 57), f'{stamp:%Y-%m-%d %H:%M}  |  forecast step {frame+1}/18', fill='black', font=font)
        draw.text((55, 85), 'Raw NC grid truth (not AWS gauges) / rainfall in mm per 10 minutes', fill='#555555', font=small)
        draw.line((55, 120, 90, 120), fill='#222222', width=3)
        draw.text((100, 109), 'Truth', fill='#222222', font=font)
        draw.line((210, 120, 245, 120), fill='#0077bb', width=3)
        draw.text((255, 109), 'Corrected forecast', fill='#0077bb', font=font)
        draw.rectangle((480, 109, 510, 129), fill='#e0f0e4')
        draw.text((520, 109), 'Truth +/- 50%', fill='#326b43', font=font)
        for j, (points, ymax) in enumerate(zip(series, maxima)):
            top, bottom = 194+j*278, 384+j*278
            def xy(index, value):
                return (left+(right-left)*index/17, bottom-(bottom-top)*value/ymax)
            draw.text((left, top-34), f'Station {records[j]["station_id"]}', fill='black', font=font)
            for tick in range(5):
                value = ymax*tick/4
                y = xy(0, value)[1]
                draw.line((left, y, right, y), fill='#dddddd')
                draw.text((left-12, y), f'{value:.2f}', fill='#444444', font=small, anchor='rm')
            if frame > 0:
                polygon = [xy(k, points[k][0]*1.5) for k in range(frame+1)]
                polygon += [xy(k, points[k][0]*.5) for k in range(frame, -1, -1)]
                draw.polygon(polygon, fill='#e0f0e4')
            for component, color in [(0, '#222222'), (1, '#0077bb')]:
                positions = [xy(k, points[k][component]) for k in range(frame+1)]
                if len(positions) > 1:
                    draw.line(positions, fill=color, width=3)
                for x, y in positions:
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
            draw.line((left, top, left, bottom, right, bottom), fill='black', width=1)
            for k in (0, 3, 6, 9, 12, 15, 17):
                x = xy(k, 0)[0]
                draw.line((x, bottom, x, bottom+5), fill='black')
                draw.text((x, bottom+10), timestamps[k].strftime('%H:%M'), fill='#444444', font=small, anchor='mt')
            draw.text((left, bottom+37), 'Forecast time', fill='#555555', font=small)
        image.save(folder / f'graph_{stamp:%Y%m%d_%H%M}.png', format='PNG')
    return folder

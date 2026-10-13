"""TC9 progressive station plots from records only; no observation file access."""
from datetime import datetime, timedelta
import math
from pathlib import Path

from PIL import Image, ImageDraw
from src.station_graphs import _font


def _runs(points, stop):
    """Consecutive valid indices, never bridging excluded timestamps."""
    run = []
    for index in range(stop + 1):
        if points[index] is None:
            if run:
                yield run
                run = []
        else:
            run.append(index)
    if run:
        yield run


def export_tc_graphs(result, outroot):
    """Save eighteen 1200x1050 plots with TC9 allowance and excluded gaps."""
    if result.get('status') != 'available':
        return None
    records = result.get('rows')
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError('Graphs require three evaluation station records')
    if [str(r.get('station_id')) for r in records] != ['410', '418', '889']:
        raise ValueError('Graph station order must be 410, 418, 889')
    series, maxima, timestamps = [], [], None
    for record in records:
        rows = record.get('rows')
        if not isinstance(rows, list) or len(rows) != 18:
            raise ValueError('Graphs require 18 rows per station')
        times = [datetime.fromisoformat(row['timestamp']) for row in rows]
        if any(t != times[0] + timedelta(minutes=10*i) for i, t in enumerate(times)):
            raise ValueError('Graph timestamps must be consecutive ten-minute intervals')
        if timestamps is not None and times != timestamps:
            raise ValueError('Station timestamps must match')
        timestamps = times
        points = []
        for row in rows:
            if row.get('excluded', False):
                points.append(None)
                continue
            try:
                truth, predicted = float(row['truth_mm10']), float(row['prediction_mm10'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('Valid graph pairs need numeric rainfall') from error
            if not all(math.isfinite(v) and v >= 0 for v in (truth, predicted)):
                raise ValueError('Valid graph rainfall must be finite and nonnegative')
            allowance = max(1., .5*truth)
            points.append((truth, predicted, max(0., truth-allowance), truth+allowance))
        series.append(points)
        maxima.append(max([1.] + [max(p[1], p[3])*1.12 for p in points if p is not None]))
    folder = Path(outroot) / 'graphs'
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError('Refusing to overwrite TC9 graphs: ' + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    title_font, font, small = _font(25), _font(18), _font(15)
    left, right = 100, 1100
    for frame, stamp in enumerate(timestamps):
        image = Image.new('RGB', (1200, 1050), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((55, 20), 'TC9 rainfall: reference and corrected forecast', fill='black', font=title_font)
        draw.text((55, 57), f'{stamp:%Y-%m-%d %H:%M} | forecast step {frame+1}/18', fill='black', font=font)
        draw.text((55, 85), 'NC grid reference, not AWS gauges | mm per 10 minutes | excluded pairs are gaps', fill='#555555', font=small)
        draw.line((55, 120, 90, 120), fill='#222222', width=3)
        draw.text((100, 109), 'Reference', fill='#222222', font=font)
        draw.line((230, 120, 265, 120), fill='#0077bb', width=3)
        draw.text((275, 109), 'Corrected forecast', fill='#0077bb', font=font)
        draw.rectangle((490, 109, 520, 129), fill='#e0f0e4')
        draw.text((530, 109), 'Allowed error: max(1.0, 50% of reference)', fill='#326b43', font=font)
        for j, (points, ymax) in enumerate(zip(series, maxima)):
            top, bottom = 194+j*278, 384+j*278
            def xy(index, value):
                return (left+(right-left)*index/17, bottom-(bottom-top)*value/ymax)
            excluded = sum(p is None for p in points[:frame+1])
            draw.text((left, top-34), f'AWS {records[j]["station_id"]} | excluded {excluded}/{frame+1} displayed pairs', fill='black', font=font)
            for tick in range(5):
                value = ymax*tick/4
                y = xy(0, value)[1]
                draw.line((left, y, right, y), fill='#dddddd')
                draw.text((left-12, y), f'{value:.2f}', fill='#444444', font=small, anchor='rm')
            runs = list(_runs(points, frame))
            for run in runs:
                if len(run) > 1:
                    polygon = [xy(k, points[k][3]) for k in run]
                    polygon += [xy(k, points[k][2]) for k in reversed(run)]
                    draw.polygon(polygon, fill='#e0f0e4')
                else:
                    k = run[0]
                    x, upper = xy(k, points[k][3])
                    lower = xy(k, points[k][2])[1]
                    draw.rectangle((x-2, upper, x+2, lower), fill='#e0f0e4')
                for component, color in [(0, '#222222'), (1, '#0077bb')]:
                    positions = [xy(k, points[k][component]) for k in run]
                    if len(positions) > 1:
                        draw.line(positions, fill=color, width=3)
                    for x, y in positions:
                        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
            if not runs:
                draw.text((left+360, top+75), 'No valid pairs', fill='#666666', font=font)
            draw.line((left, top, left, bottom, right, bottom), fill='black', width=1)
            for k in (0, 3, 6, 9, 12, 15, 17):
                x = xy(k, 0)[0]
                draw.line((x, bottom, x, bottom+5), fill='black')
                draw.text((x, bottom+10), timestamps[k].strftime('%H:%M'), fill='#444444', font=small, anchor='mt')
            draw.text((left, bottom+37), 'Forecast time', fill='#555555', font=small)
        image.save(folder / f'graph_{stamp:%Y%m%d_%H%M}.png', format='PNG')
    return folder

from typing import List
import pandas as pd

def windowize(input_df, window_len, model_centre_focus_len, get_fill_seq):

    assert (window_len - model_centre_focus_len) % 2 == 0, "window_len - model_centre_focus_len must be even, for even padding on both sides"
    window_padding_each_side = (window_len - model_centre_focus_len) / 2

    def get_windows(seq: str, n_windows: int) -> List[str]:
        windows = []
        # mid index, on even floor mid
        mid = len(seq) // 2 if len(seq) % 2 == 1 else len(seq) // 2 - 1
        focus_mid_start = mid - ((model_centre_focus_len / 2) - 1) if model_centre_focus_len % 2 == 0 else mid - (model_centre_focus_len // 2)
        for i in range(n_windows):
            # calculate theoretical start and end
            if i % 2 == 0:
                start = focus_mid_start - (i // 2) * model_centre_focus_len - window_padding_each_side
                end = focus_mid_start - (i // 2) * model_centre_focus_len + model_centre_focus_len + window_padding_each_side
            else:
                start = focus_mid_start + (i // 2 + 1) * model_centre_focus_len - window_padding_each_side
                end = focus_mid_start + (i // 2 + 1) * model_centre_focus_len + model_centre_focus_len + window_padding_each_side
            start, end = int(start), int(end)

            # if theoretical start and end are out of bounds, pad
            if start < 0 and end > len(seq):
                window = get_fill_seq(-start) + seq + get_fill_seq(end - len(seq))
            elif start < 0:
                window = get_fill_seq(-start) + seq[:end]
            elif end > len(seq):
                window = seq[start:] + get_fill_seq(end - len(seq))
            else:
                window = seq[start:end]
            windows.append(window)
        return windows

    def windowizer(df: pd.DataFrame) -> pd.DataFrame:
        new_rows = []
        for _, row in df.iterrows():
            seq = row['seq']
            assert len(seq) >= model_centre_focus_len, "Sequence is shorter than model_centre_focus_len"
            n_windows_precalculation = len(seq) // model_centre_focus_len
            n_windows = n_windows_precalculation - 1 if n_windows_precalculation % 2 == 0 else n_windows_precalculation
            windows = get_windows(seq, n_windows)
            for i, window in enumerate(windows, start=1):
                new_row = row.copy()
                new_row['window'] = window
                new_row['window_index'] = i
                new_rows.append(new_row)
        new_df = pd.DataFrame(new_rows)
        return new_df

    return windowizer(input_df)
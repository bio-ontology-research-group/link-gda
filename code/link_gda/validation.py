"""Validation eligibility fixed from annotations and training associations."""
def eligible_validation(validation, training, candidates, scope='all'):
    selected=validation[validation.Gene.isin(candidates)].copy()
    if scope=='unseen-gene':selected=selected[~selected.Gene.isin(set(training.Gene))].copy()
    elif scope!='all':raise ValueError('Unknown validation scope')
    if selected.Disease.nunique()<2:raise ValueError('Fewer than two eligible validation diseases')
    return selected

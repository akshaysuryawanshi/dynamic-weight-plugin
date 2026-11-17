"""Request consistency is independent of the qualified loading strategy."""

from .control import ControlError
def loading_plan(model_type, current, target, mode):
    if mode not in {"drain", "dirty"}:
        raise ControlError("invalid_request", "mode must be drain or dirty; read capabilities")
    from .representations import REPRESENTATIONS
    for name in (current, target):
        representation = REPRESENTATIONS.get(name)
        if representation is None:
            raise ControlError("invalid_request", "Unknown representation: " + name)
        if mode not in representation.modes:
            allowed = " or ".join(representation.modes)
            raise ControlError("invalid_request", f"{name} only supports {allowed} mode")
    from .compat import runtime_profile
    profile = runtime_profile()
    if {current, target} <= set(profile.get('component_formats', ())):
        return {'strategy': 'native_components',
                'scheduling': 'between_steps' if mode == 'dirty' else 'uninterrupted',
                'consistency': 'mixed_weights_and_existing_kv' if mode == 'dirty' else 'drained_requests',
                'rollback_available': False, 'selection_reason': 'native_module_loading_contract'}
    raise ControlError('invalid_request', 'Representation pair has no native component loader adapter')

"""Single-rank native expert banks; routing and forward owners stay live."""

from copy import copy


def validate_experts(model, path, old):
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    parent_path, name = path.rsplit('.', 1)
    parent = model.get_submodule(parent_path)
    cfg = old.moe_config
    parallel = cfg.moe_parallel_config
    manager = old.expert_map_manager
    checks = {
        'runner_type': type(parent) is MoERunner,
        'child_name': name == 'routed_experts',
        'child_owner': parent.routed_experts is old,
        'config_owner': parent.moe_config is cfg,
        'shared_experts': parent.shared_experts is None and parent.shared_expert_gate is None,
        'dbo': not parent.enable_dbo,
        'transforms': parent.routed_input_transform is None and parent.routed_output_transform is None,
        'lora': not cfg.is_lora_enabled,
        'hash_routing': not cfg.has_hash_routing,
        'shared_prefix': not cfg.shared_expert_prefix,
        'deferred': not cfg._defer_moe_finalize,
        'bias': not cfg.has_bias,
        'parallel': all(getattr(parallel, n) == 1 for n in ('tp_size','ep_size','dp_size','pcp_size','sp_size')),
        'eplb': not parallel.enable_eplb and not parallel.use_ep,
        'expert_maps': not manager.num_fused_shared_experts and manager.expert_map is None
                       and manager.expert_mask is None and manager.routing_tables is None,
        'routing_hook': old.routing_sink is None and old.custom_routing_function is None,
        'correction_bias': old.e_score_correction_bias is None,
        # UnquantizedFusedMoEMethod is also a CustomOp/Module. Its complete
        # subtree belongs to the bank and is retired with it.
        'children': all(name == 'quant_method' and child is old.quant_method
                        for name, child in old.named_children()),
    }
    failed = [key for key, valid in checks.items() if not valid]
    if failed:
        raise ValueError('Unreviewed routed-expert ownership/configuration: ' + path + ': ' + ', '.join(failed))



def constructor_args(old, quant_config):
    # Native constructors pad these dimensions in place. Start from canonical
    # logical sizes every time and never let them mutate the serving config.
    cfg = copy(old.moe_config)
    cfg.moe_parallel_config = copy(cfg.moe_parallel_config)
    cfg.hidden_dim = cfg.hidden_dim_unpadded
    cfg.intermediate_size_per_partition = cfg.intermediate_size_per_partition_unpadded
    names = ('layer_name', 'params_dtype', 'ckpt_gate_proj_name', 'ckpt_down_proj_name',
             'ckpt_up_proj_name', 'is_fused_checkpoint_transposed', 'renormalize',
             'use_grouped_topk', 'num_expert_group', 'topk_group',
             'custom_routing_function', 'scoring_func', 'routed_scaling_factor',
             'swiglu_limit', 'swiglu_alpha', 'swiglu_beta', 'e_score_correction_bias',
             'apply_router_weight_on_input')
    return {**{name: getattr(old, name) for name in names},
            'moe_config': cfg, 'quant_config': quant_config,
            'expert_map_manager': copy(old.expert_map_manager)}


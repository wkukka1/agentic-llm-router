import sys, warnings
sys.path.insert(0, "src")
import pandas as pd
from router.routing.routers import MatrixRouter
from router.routing.base import UnsupportedCandidatePolicy
from router.router import Router
from router.context import RoutingRequest
from router.constraints import RoutingConstraints, OptimizationObjective
from router.llm.registry import LLMRegistry
from router.llm.profile import LLMProfile
from router.models.registry import RouterModelFactory

class TextMatrix(MatrixRouter):
    can_route_text = True
    def predict_scores_text(self, texts, *, encoder=None):
        return pd.DataFrame([[0.80, 0.70]] * len(texts), columns=self.model_ids)

scores = pd.DataFrame([[0.8, 0.7]], index=["q"], columns=["big", "small"])
rm = TextMatrix(scores, model_costs=[0.010, 0.001])          # USD / query
rm.unsupported_policy = UnsupportedCandidatePolicy.SCORE_WITH_PRIOR

reg = LLMRegistry()
for mid, out_cost in [("big", 6e-5), ("small", 6e-6), ("unseen", 6e-5)]:
    reg.register(LLMProfile(name=mid, provider="x", model_id=mid, output_cost_per_token=out_cost))

r = Router(rm, llm_registry=reg)
req = RoutingRequest("r1", "hi", constraints=RoutingConstraints(objective=OptimizationObjective(cost_weight=10.0)))
dec = r.route(req)
print("== ROUTER-001 cost-scale mix ==")
for s in dec.ranked_models:
    print(f"{s.model.model_id:8s} supported={s.supported!s:5s} expected_cost={s.expected_cost:<10g} quality={s.expected_quality:.3f}")
print("winner:", dec.ranked_models[0].model.model_id)

print("== REGISTRY-001 cost_model kwarg ==")
from types import SimpleNamespace
class Dummy:
    @classmethod
    def from_artifact(cls, artifact, cost_model=None):
        return cost_model
f = RouterModelFactory(); f.register("dummy", Dummy)
art = SimpleNamespace(router_model_name="dummy")
print("positional:", f.create(art, "CM"), "| keyword:", f.create(art, cost_model="CM"))

print("== ROUTERMODEL-001 __call__ bound to base route ==")
class Loud(MatrixRouter):
    def route(self, *a, **k):
        raise RuntimeError("subclass route() called")
m = Loud(scores)
try: m.route(["q"])
except RuntimeError as e: print("m.route ->", e)
print("m(['q']) ->", type(m(["q"])).__name__, "(subclass override bypassed)")

from router.routing import NIRTRouter
from router.agentic import AgenticRouter

router = NIRTRouter.from_run("nirt-2d-projected")
result = router.route_text(["Prove that sqrt(2) is irrational."])
result.selected_model_ids, result.model_mix()

agent = AgenticRouter(router)
outcome = agent.run("Plan a 3-day trip to Kyoto and translate the itinerary to Japanese.")
print(outcome.mode)
print(outcome.answer)
print(outcome.cost)
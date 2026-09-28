from types import SimpleNamespace

from verl.trainer import agentopsd_cleanup


def test_cleanup_runtime_resources_closes_envs_kills_unique_workers_and_placement_groups(monkeypatch):
    events = []

    class Handle:
        def __init__(self, actor_id):
            self._actor_id = actor_id

    shared_handle = Handle("shared")
    second_handle = Handle("second")

    class FakeRay:
        @staticmethod
        def kill(handle, no_restart=True):
            events.append(("kill", handle._actor_id, no_restart))

    class FakeEnvironment:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(("close", self.name))

    class FakeWorkerGroup:
        def __init__(self, workers):
            self.workers = workers

    class FakePlacementGroup:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        agentopsd_cleanup,
        "remove_placement_group",
        lambda placement_group: events.append(("remove_pg", placement_group.name)),
    )

    trainer = SimpleNamespace(
        actor_rollout_wg=FakeWorkerGroup([shared_handle, second_handle]),
        critic_wg=FakeWorkerGroup([shared_handle]),
        ref_policy_wg=None,
        rm_wg=None,
        _worker_groups=[FakeWorkerGroup([Handle("partial")])],
        resource_pool_manager=SimpleNamespace(
            resource_pool_dict={
                "global": SimpleNamespace(
                    pgs=[FakePlacementGroup("global-0"), FakePlacementGroup("global-1")]
                )
            }
        ),
    )

    agentopsd_cleanup.cleanup_runtime_resources(
        trainer=trainer,
        envs=[FakeEnvironment("train"), FakeEnvironment("val")],
        ray_module=FakeRay,
    )

    assert events == [
        ("close", "train"),
        ("close", "val"),
        ("kill", "shared", True),
        ("kill", "second", True),
        ("kill", "partial", True),
        ("remove_pg", "global-0"),
        ("remove_pg", "global-1"),
    ]



def test_webshop_close_is_idempotent_and_kills_workers(monkeypatch):
    from agent_system.environments.env_package.webshop import envs as webshop_envs

    events = []

    class RemoteClose:
        def __init__(self, name):
            self.name = name

        def remote(self):
            events.append(("close", self.name))
            return self.name

    class Worker:
        def __init__(self, name):
            self.name = name
            self.close = RemoteClose(name)

    wrapper = object.__new__(webshop_envs.WebshopMultiProcessEnv)
    wrapper._workers = [Worker("a"), Worker("b")]
    wrapper._closed = False
    monkeypatch.setattr(
        webshop_envs.ray,
        "get",
        lambda refs, timeout=None: events.append(("get", tuple(refs))),
    )
    monkeypatch.setattr(
        webshop_envs.ray,
        "kill",
        lambda worker, no_restart=True: events.append(("kill", worker.name)),
    )

    wrapper.close()
    wrapper.close()

    assert events == [
        ("close", "a"),
        ("close", "b"),
        ("get", ("a", "b")),
        ("kill", "a"),
        ("kill", "b"),
    ]
    assert wrapper._workers == []

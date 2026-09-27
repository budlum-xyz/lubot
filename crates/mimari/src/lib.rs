//! Layered component wiring, with the layer rule enforced where it is violated.
//!
//! # The rule
//!
//! A component may depend on components in its own layer or below, never above.
//! That is the whole content of "layered architecture", and it is the invariant
//! that erodes: one upward call is invisible at the call site and turns a layer
//! into a knot that cannot be started, replaced, or tested on its own.
//!
//! [`Blueprint::connect`] refuses the upward edge at the moment it is written,
//! with both layers named. The alternative - discover it when the start order
//! comes out wrong, or never - means the erosion is only visible once the system
//! will not boot.
//!
//! # Start order is checked, not trusted
//!
//! [`Blueprint::start_order`] produces a topological order and then verifies it
//! against every declared edge. A topological sort that is subtly wrong produces
//! a plausible-looking order that starts a component before what it needs, and
//! the failure is a nil dereference somewhere unrelated. Verifying costs one pass
//! over the edges and turns a silent ordering bug into
//! [`WiringError::OrderViolatesEdges`].
//!
//! # Shutdown is the exact reverse
//!
//! [`Blueprint::shutdown_order`] is the start order reversed, not a separately
//! computed order. A component may be used by something started after it, so the
//! only safe teardown is the mirror of the build. A second, independently
//! computed order is a second place to be wrong.
//!
//! # Registered is not started
//!
//! [`Blueprint::mark_started`] is a separate call from registration, and
//! [`Blueprint::ready`] reports which components are both started and have all
//! their dependencies started. A component that exists in the graph but never
//! came up looks identical to one that did unless the distinction is recorded.

use std::collections::{BTreeMap, BTreeSet};

/// Why the blueprint refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WiringError {
    /// The component name is taken.
    DuplicateComponent { name: String },
    /// No component with this name.
    UnknownComponent { name: String },
    /// A component cannot depend on itself.
    SelfDependency { name: String },
    /// The edge points upward, which is the rule being enforced.
    UpwardDependency {
        from: String,
        from_layer: u32,
        to: String,
        to_layer: u32,
    },
    /// The edge would close a cycle. Same-layer edges can still do this.
    Cycle { from: String, to: String },
    /// The computed order does not respect a declared edge. Reaching this means
    /// the sort itself is wrong, which is why it is checked rather than assumed.
    OrderViolatesEdges { before: String, after: String },
}

impl std::fmt::Display for WiringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateComponent { name } => write!(f, "the component {name:?} already exists"),
            Self::UnknownComponent { name } => write!(f, "there is no component {name:?}"),
            Self::SelfDependency { name } => write!(f, "component {name:?} cannot depend on itself"),
            Self::UpwardDependency {
                from,
                from_layer,
                to,
                to_layer,
            } => write!(
                f,
                "{from:?} is in layer {from_layer} and cannot depend on {to:?} in layer {to_layer}; an upward call turns a layer into a knot"
            ),
            Self::Cycle { from, to } => write!(
                f,
                "making {to:?} depend on {from:?} would close a cycle, and no start order can satisfy it"
            ),
            Self::OrderViolatesEdges { before, after } => write!(
                f,
                "the computed order starts {after:?} before {before:?}, which depends on it"
            ),
        }
    }
}

/// One component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub name: String,
    /// Lower is deeper. Dependencies must be at this number or below.
    pub layer: u32,
    /// What it depends on.
    pub dependencies: BTreeSet<String>,
    /// Whether it has come up. Registered is not started, and the difference is
    /// the whole content of "the system is up".
    pub started: bool,
}

/// The blueprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blueprint {
    components: BTreeMap<String, Component>,
}

impl Default for Blueprint {
    fn default() -> Self {
        Self::new()
    }
}

impl Blueprint {
    /// An empty blueprint.
    #[must_use]
    pub fn new() -> Self {
        Self {
            components: BTreeMap::new(),
        }
    }

    /// Adds a component at a layer.
    ///
    /// # Errors
    ///
    /// [`WiringError::DuplicateComponent`].
    pub fn add(&mut self, name: &str, layer: u32) -> Result<(), WiringError> {
        if self.components.contains_key(name) {
            return Err(WiringError::DuplicateComponent {
                name: name.to_string(),
            });
        }
        self.components.insert(
            name.to_string(),
            Component {
                name: name.to_string(),
                layer,
                dependencies: BTreeSet::new(),
                started: false,
            },
        );
        Ok(())
    }

    /// Declares that `from` depends on `to`.
    ///
    /// # Errors
    ///
    /// Any [`WiringError`] that applies.
    pub fn connect(&mut self, from: &str, to: &str) -> Result<(), WiringError> {
        if from == to {
            return Err(WiringError::SelfDependency {
                name: from.to_string(),
            });
        }
        let (from_layer, to_layer) = match (self.components.get(from), self.components.get(to)) {
            (Some(a), Some(b)) => (a.layer, b.layer),
            (None, _) => {
                return Err(WiringError::UnknownComponent {
                    name: from.to_string(),
                })
            }
            (_, None) => {
                return Err(WiringError::UnknownComponent {
                    name: to.to_string(),
                })
            }
        };
        // The layer rule, enforced where it is violated rather than where the
        // damage shows up.
        if to_layer > from_layer {
            return Err(WiringError::UpwardDependency {
                from: from.to_string(),
                from_layer,
                to: to.to_string(),
                to_layer,
            });
        }
        if self.reaches(to, from) {
            return Err(WiringError::Cycle {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        if let Some(component) = self.components.get_mut(from) {
            component.dependencies.insert(to.to_string());
        }
        Ok(())
    }

    /// Whether `target` is reachable from `start` by following dependencies.
    fn reaches(&self, start: &str, target: &str) -> bool {
        let mut stack = vec![start.to_string()];
        let mut seen = BTreeSet::new();
        while let Some(current) = stack.pop() {
            if !seen.insert(current.clone()) {
                continue;
            }
            let Some(component) = self.components.get(&current) else {
                continue;
            };
            for dependency in &component.dependencies {
                if dependency == target {
                    return true;
                }
                stack.push(dependency.clone());
            }
        }
        false
    }

    /// Marks a component started.
    ///
    /// # Errors
    ///
    /// [`WiringError::UnknownComponent`].
    pub fn mark_started(&mut self, name: &str) -> Result<(), WiringError> {
        let Some(component) = self.components.get_mut(name) else {
            return Err(WiringError::UnknownComponent {
                name: name.to_string(),
            });
        };
        component.started = true;
        Ok(())
    }

    /// Marks a component stopped.
    ///
    /// # Errors
    ///
    /// [`WiringError::UnknownComponent`].
    pub fn mark_stopped(&mut self, name: &str) -> Result<(), WiringError> {
        let Some(component) = self.components.get_mut(name) else {
            return Err(WiringError::UnknownComponent {
                name: name.to_string(),
            });
        };
        component.started = false;
        Ok(())
    }

    /// The order to start components in, verified against the declared edges.
    ///
    /// # Errors
    ///
    /// [`WiringError::OrderViolatesEdges`] if the sort produced an order that
    /// contradicts an edge. This should be unreachable; it is checked because a
    /// subtly wrong order starts a component before what it needs and fails
    /// somewhere unrelated.
    pub fn start_order(&self) -> Result<Vec<String>, WiringError> {
        // Kahn's algorithm over a deterministic key order, so the same blueprint
        // always produces the same order.
        let mut remaining: BTreeMap<&str, BTreeSet<&str>> = self
            .components
            .iter()
            .map(|(name, c)| {
                (
                    name.as_str(),
                    c.dependencies.iter().map(String::as_str).collect(),
                )
            })
            .collect();
        let mut order = Vec::with_capacity(remaining.len());
        while !remaining.is_empty() {
            let ready: Vec<&str> = remaining
                .iter()
                .filter(|(_, deps)| deps.is_empty())
                .map(|(name, _)| *name)
                .collect();
            // A cycle would leave nothing ready while components remain. The
            // blueprint refuses cyclic edges, so this is a guard against a bug
            // here rather than against bad input; breaking out would silently
            // produce a short order.
            if ready.is_empty() {
                let first = remaining
                    .keys()
                    .next()
                    .map_or(String::new(), ToString::to_string);
                return Err(WiringError::Cycle {
                    from: first.clone(),
                    to: first,
                });
            }
            for name in &ready {
                remaining.remove(*name);
            }
            for name in &ready {
                order.push((*name).to_string());
            }
            for deps in remaining.values_mut() {
                for name in &ready {
                    deps.remove(*name);
                }
            }
        }
        // Verify rather than trust: an order that starts a component before what
        // it depends on is worse than no order at all.
        let position: BTreeMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        for (name, component) in &self.components {
            for dependency in &component.dependencies {
                let after = position.get(name.as_str()).copied().unwrap_or(0);
                let before = position.get(dependency.as_str()).copied().unwrap_or(0);
                if before >= after {
                    return Err(WiringError::OrderViolatesEdges {
                        before: dependency.clone(),
                        after: name.clone(),
                    });
                }
            }
        }
        Ok(order)
    }

    /// The order to stop components in: the start order reversed.
    ///
    /// Not a separately computed order. A component may be used by something
    /// started after it, so the only safe teardown is the mirror of the build; a
    /// second independent computation is a second place to be wrong.
    ///
    /// # Errors
    ///
    /// Propagates from [`Self::start_order`].
    pub fn shutdown_order(&self) -> Result<Vec<String>, WiringError> {
        let mut order = self.start_order()?;
        order.reverse();
        Ok(order)
    }

    /// The components that are started and whose dependencies are all started.
    ///
    /// "The system is up" is not "the graph is populated".
    #[must_use]
    pub fn ready(&self) -> Vec<&str> {
        self.components
            .values()
            .filter(|c| {
                c.started
                    && c.dependencies
                        .iter()
                        .all(|d| self.components.get(d).is_some_and(|dep| dep.started))
            })
            .map(|c| c.name.as_str())
            .collect()
    }

    /// Reads a component.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Component> {
        self.components.get(name)
    }

    /// The distinct layers in use, in ascending order.
    #[must_use]
    pub fn layers(&self) -> Vec<u32> {
        let mut layers: Vec<u32> = self.components.values().map(|c| c.layer).collect();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    /// How many components are declared.
    #[must_use]
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether the blueprint is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// storage (0) <- domain (1) <- api (2).
    fn layered() -> Blueprint {
        let mut b = Blueprint::new();
        b.add("storage", 0).expect("storage");
        b.add("domain", 1).expect("domain");
        b.add("api", 2).expect("api");
        b
    }

    #[test]
    fn a_downward_dependency_is_allowed() {
        let mut b = layered();
        assert!(b.connect("domain", "storage").is_ok());
        assert!(b.connect("api", "domain").is_ok());
    }

    #[test]
    fn an_upward_dependency_is_refused_with_both_layers_named() {
        // One upward call is invisible at the call site and turns a layer into a
        // knot that cannot be started or replaced on its own.
        let mut b = layered();
        assert_eq!(
            b.connect("storage", "api"),
            Err(WiringError::UpwardDependency {
                from: "storage".to_string(),
                from_layer: 0,
                to: "api".to_string(),
                to_layer: 2,
            })
        );
    }

    #[test]
    fn a_same_layer_dependency_is_allowed() {
        let mut b = Blueprint::new();
        b.add("left", 1).expect("left");
        b.add("right", 1).expect("right");
        assert!(b.connect("left", "right").is_ok());
    }

    #[test]
    fn a_same_layer_cycle_is_refused() {
        // Same-layer edges are legal, so they are the only way to build a cycle
        // once the layer rule holds.
        let mut b = Blueprint::new();
        b.add("left", 1).expect("left");
        b.add("right", 1).expect("right");
        b.connect("left", "right").expect("edge");
        assert_eq!(
            b.connect("right", "left"),
            Err(WiringError::Cycle {
                from: "right".to_string(),
                to: "left".to_string(),
            })
        );
    }

    #[test]
    fn a_component_cannot_depend_on_itself() {
        let mut b = layered();
        assert_eq!(
            b.connect("domain", "domain"),
            Err(WiringError::SelfDependency {
                name: "domain".to_string()
            })
        );
    }

    #[test]
    fn connecting_an_unknown_component_is_refused() {
        let mut b = layered();
        assert_eq!(
            b.connect("domain", "ghost"),
            Err(WiringError::UnknownComponent {
                name: "ghost".to_string()
            })
        );
    }

    #[test]
    fn the_start_order_puts_dependencies_first() {
        let mut b = layered();
        b.connect("api", "domain").expect("edge");
        b.connect("domain", "storage").expect("edge");
        assert_eq!(
            b.start_order(),
            Ok(vec![
                "storage".to_string(),
                "domain".to_string(),
                "api".to_string(),
            ])
        );
    }

    #[test]
    fn the_shutdown_order_is_the_exact_reverse() {
        // Not a separately computed order: a component may be used by something
        // started after it, so the only safe teardown is the mirror of the build.
        let mut b = layered();
        b.connect("api", "domain").expect("edge");
        b.connect("domain", "storage").expect("edge");
        let start = b.start_order().expect("start");
        let stop = b.shutdown_order().expect("stop");
        let mut expected = start.clone();
        expected.reverse();
        assert_eq!(stop, expected);
        assert_eq!(stop.first().map(String::as_str), Some("api"));
        assert_eq!(stop.last().map(String::as_str), Some("storage"));
    }

    #[test]
    fn the_order_is_verified_against_every_edge() {
        // A subtly wrong topological sort starts a component before what it needs
        // and fails somewhere unrelated. Checking costs one pass.
        let mut b = Blueprint::new();
        for name in ["a", "b", "c", "d"] {
            b.add(name, 0).expect("add");
        }
        b.connect("d", "c").expect("edge");
        b.connect("c", "b").expect("edge");
        b.connect("b", "a").expect("edge");
        let order = b.start_order().expect("order");
        assert_eq!(order, vec!["a", "b", "c", "d"]);
        let position: BTreeMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        for (name, component) in [("d", "c"), ("c", "b"), ("b", "a")] {
            assert!(position[component] < position[name]);
        }
    }

    #[test]
    fn a_registered_component_is_not_started() {
        // "The system is up" is not "the graph is populated".
        let mut b = layered();
        b.connect("domain", "storage").expect("edge");
        assert!(
            b.ready().is_empty(),
            "nothing was started but ready was not empty"
        );
        b.mark_started("storage").expect("start storage");
        assert_eq!(b.ready(), vec!["storage"]);
        assert!(
            !b.ready().contains(&"domain"),
            "domain was reported ready before it was started"
        );
    }

    #[test]
    fn a_started_component_with_an_unstarted_dependency_is_not_ready() {
        let mut b = layered();
        b.connect("domain", "storage").expect("edge");
        b.mark_started("domain").expect("start domain");
        assert!(b.ready().is_empty());
        b.mark_started("storage").expect("start storage");
        assert_eq!(b.ready().len(), 2);
    }

    #[test]
    fn stopping_a_dependency_makes_its_dependents_unready() {
        let mut b = layered();
        b.connect("domain", "storage").expect("edge");
        b.mark_started("storage").expect("start");
        b.mark_started("domain").expect("start");
        assert_eq!(b.ready().len(), 2);
        b.mark_stopped("storage").expect("stop");
        // Both are now unready: storage because it is stopped, domain because the
        // component it depends on is stopped.
        assert!(b.ready().is_empty());
        assert!(b.get("storage").is_some_and(|c| !c.started));
        assert!(b.get("domain").is_some_and(|c| c.started));
    }

    #[test]
    fn layers_are_reported_in_order_without_duplicates() {
        let mut b = Blueprint::new();
        b.add("a", 2).expect("a");
        b.add("b", 0).expect("b");
        b.add("c", 0).expect("c");
        assert_eq!(b.layers(), vec![0, 2]);
    }

    #[test]
    fn a_duplicate_component_is_refused() {
        let mut b = layered();
        assert_eq!(
            b.add("api", 0),
            Err(WiringError::DuplicateComponent {
                name: "api".to_string()
            })
        );
    }

    #[test]
    fn marking_an_unknown_component_is_refused() {
        let mut b = layered();
        assert_eq!(
            b.mark_started("ghost"),
            Err(WiringError::UnknownComponent {
                name: "ghost".to_string()
            })
        );
        assert_eq!(
            b.mark_stopped("ghost"),
            Err(WiringError::UnknownComponent {
                name: "ghost".to_string()
            })
        );
    }

    #[test]
    fn an_empty_blueprint_has_an_empty_order() {
        let b = Blueprint::new();
        assert!(b.is_empty());
        assert_eq!(b.start_order(), Ok(Vec::new()));
        assert!(b.ready().is_empty());
    }

    #[test]
    fn the_same_blueprint_always_produces_the_same_order() {
        // Deterministic, so a boot that works twice is not luck.
        let mut b = Blueprint::new();
        for name in ["a", "b", "c"] {
            b.add(name, 0).expect("add");
        }
        let first = b.start_order().expect("order");
        let second = b.start_order().expect("order");
        assert_eq!(first, second);
        assert_eq!(first, vec!["a", "b", "c"]);
    }
}

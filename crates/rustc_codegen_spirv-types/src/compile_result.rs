use rspirv::dr::Operand;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModuleResult {
    SingleModule(PathBuf),
    MultiModule(BTreeMap<EntryPoint, PathBuf>),
}

impl ModuleResult {
    pub fn unwrap_single(&self) -> &Path {
        match self {
            ModuleResult::SingleModule(result) => result,
            ModuleResult::MultiModule(_) => {
                panic!("called `ModuleResult::unwrap_single()` on a `MultiModule` result")
            }
        }
    }

    pub fn unwrap_multi(&self) -> &BTreeMap<EntryPoint, PathBuf> {
        match self {
            ModuleResult::MultiModule(result) => result,
            ModuleResult::SingleModule(_) => {
                panic!("called `ModuleResult::unwrap_multi()` on a `SingleModule` result")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, PartialOrd, Ord, Hash)]
pub struct EntryPoint {
    pub execution_model: spirv::ExecutionModel,
    // IdRef
    pub entry_point: spirv::Word,
    // LiteralString
    pub name: String,
    // Vec<IdRef>
    pub interface: Vec<spirv::Word>,
}

#[derive(Debug, thiserror::Error)]
pub enum EntryPointConversionError {
    #[error("Operand was not execution model.\nExpected: ExecutionModel. Found: {0}")]
    MissingExecutionModel(Operand),
    #[error("Entry point id was not a 32bit literal.\nExpected: IdRef. Found: {0}")]
    MissingEntryPoint(Operand),
    #[error("Entry point name was not a string literal.\nFound: {0}")]
    InvalidName(Operand),
    #[error("Interface contains invelid 32bit literal.\nFound: {0}")]
    InvalidInterface(Operand),
    #[error("Not enough operands.\nOperands: {}", 0.to_string())]
    NotEnoughOperands(Vec<Operand>),
}

impl TryFrom<Vec<Operand>> for EntryPoint {
    type Error = EntryPointConversionError;

    fn try_from(value: Vec<Operand>) -> Result<Self, Self::Error> {
        let mut iter = value.iter().cloned();
        Ok(Self {
            execution_model: match iter.next() {
                Some(op) => match op {
                    Operand::ExecutionModel(model) => model,
                    _ => return Err(EntryPointConversionError::MissingExecutionModel(op)),
                },
                None => return Err(EntryPointConversionError::NotEnoughOperands(value)),
            },
            entry_point: match iter.next() {
                Some(op) => match op {
                    Operand::IdRef(word) => word,
                    _ => return Err(EntryPointConversionError::MissingEntryPoint(op)),
                },
                None => return Err(EntryPointConversionError::NotEnoughOperands(value)),
            },
            name: match iter.next() {
                Some(op) => match op {
                    Operand::LiteralString(string) => string,
                    _ => return Err(EntryPointConversionError::InvalidName(op)),
                },
                None => return Err(EntryPointConversionError::NotEnoughOperands(value)),
            },
            interface: {
                let res: std::result::Result<Vec<spirv::Word>, _> = iter
                    .map(|op| match op {
                        Operand::IdRef(word) => Ok(word),
                        _ => Err(EntryPointConversionError::InvalidInterface(op)),
                    })
                    .collect();
                res?
            },
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompileResult {
    pub entry_points: Vec<EntryPoint>,
    pub module: ModuleResult,
}

impl CompileResult {
    pub fn codegen_entry_point_strings(&self) -> String {
        let trie = Trie::create_from(self.entry_points.iter().map(|x| &x.name as &str));
        let mut builder = String::new();
        trie.emit(&mut builder, String::new(), 0);
        builder
    }
}

#[derive(Default)]
struct Trie<'a> {
    present: bool,
    children: BTreeMap<&'a str, Trie<'a>>,
}

impl<'a> Trie<'a> {
    fn create_from(entry_points: impl IntoIterator<Item = &'a str>) -> Self {
        let mut result = Trie::default();
        for entry in entry_points {
            result.insert(entry.split("::"));
        }
        result
    }

    fn insert(&mut self, mut sequence: impl Iterator<Item = &'a str>) {
        match sequence.next() {
            None => self.present = true,
            Some(next) => self.children.entry(next).or_default().insert(sequence),
        }
    }

    fn emit(&self, builder: &mut String, full_name: String, indent: usize) {
        let mut children = self.children.iter().collect::<Vec<_>>();
        children.sort_unstable_by(|(k1, _), (k2, _)| k1.cmp(k2));
        for (child_name, child) in children {
            let full_child_name = if full_name.is_empty() {
                (*child_name).to_string()
            } else {
                format!("{full_name}::{child_name}")
            };
            if child.present {
                assert!(child.children.is_empty());
                writeln!(
                    builder,
                    "{:indent$}#[allow(non_upper_case_globals)]",
                    "",
                    indent = indent * 4
                )
                .unwrap();
                writeln!(
                    builder,
                    "{:indent$}pub const {}: &str = \"{}\";",
                    "",
                    child_name,
                    full_child_name,
                    indent = indent * 4
                )
                .unwrap();
            } else {
                writeln!(
                    builder,
                    "{:indent$}pub mod {} {{",
                    "",
                    child_name,
                    indent = indent * 4
                )
                .unwrap();
                child.emit(builder, full_child_name, indent + 1);
                writeln!(builder, "{:indent$}}}", "", indent = indent * 4).unwrap();
            }
        }
    }
}

#[allow(non_upper_case_globals)]
pub const a: &str = "x::a";

#[cfg(test)]
mod test {
    use super::*;

    fn test<const N: usize>(arr: [&str; N], expected: &str) {
        let trie = Trie::create_from(IntoIterator::into_iter(arr));
        let mut builder = String::new();
        trie.emit(&mut builder, String::new(), 0);
        assert_eq!(builder, expected);
    }

    #[test]
    fn basic() {
        test(
            ["a", "b"],
            r#"#[allow(non_upper_case_globals)]
pub const a: &str = "a";
#[allow(non_upper_case_globals)]
pub const b: &str = "b";
"#,
        );
    }

    #[test]
    fn modules() {
        test(
            ["a", "x::a", "x::b", "x::y::a", "y::z::a"],
            r#"#[allow(non_upper_case_globals)]
pub const a: &str = "a";
pub mod x {
    #[allow(non_upper_case_globals)]
    pub const a: &str = "x::a";
    #[allow(non_upper_case_globals)]
    pub const b: &str = "x::b";
    pub mod y {
        #[allow(non_upper_case_globals)]
        pub const a: &str = "x::y::a";
    }
}
pub mod y {
    pub mod z {
        #[allow(non_upper_case_globals)]
        pub const a: &str = "y::z::a";
    }
}
"#,
        );
    }
}

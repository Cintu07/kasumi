/// Built-in functions of the formula language.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Tanh,
    Atan2,
    Sqrt,
    Abs,
    Fract,
    Floor,
    Sign,
    Min,
    Max,
    Mix,
    Clamp,
    Smoothstep,
    Mag,
    Exp,
    Log,
    Pow,
    Noise,
    Rnd,
}

impl Func {
    pub const ALL: &'static [Func] = &[
        Func::Sin,
        Func::Cos,
        Func::Tan,
        Func::Tanh,
        Func::Atan2,
        Func::Sqrt,
        Func::Abs,
        Func::Fract,
        Func::Floor,
        Func::Sign,
        Func::Min,
        Func::Max,
        Func::Mix,
        Func::Clamp,
        Func::Smoothstep,
        Func::Mag,
        Func::Exp,
        Func::Log,
        Func::Pow,
        Func::Noise,
        Func::Rnd,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Tanh => "tanh",
            Func::Atan2 => "atan2",
            Func::Sqrt => "sqrt",
            Func::Abs => "abs",
            Func::Fract => "fract",
            Func::Floor => "floor",
            Func::Sign => "sign",
            Func::Min => "min",
            Func::Max => "max",
            Func::Mix => "mix",
            Func::Clamp => "clamp",
            Func::Smoothstep => "smoothstep",
            Func::Mag => "mag",
            Func::Exp => "exp",
            Func::Log => "log",
            Func::Pow => "pow",
            Func::Noise => "noise",
            Func::Rnd => "rnd",
        }
    }

    pub fn arity(self) -> usize {
        match self {
            Func::Atan2 | Func::Min | Func::Max | Func::Mag | Func::Pow | Func::Noise => 2,
            Func::Mix | Func::Clamp | Func::Smoothstep => 3,
            _ => 1,
        }
    }

    pub fn from_name(s: &str) -> Option<Func> {
        Func::ALL.iter().copied().find(|f| f.name() == s)
    }

    /// Functions the mutator may swap this one for (same arity, similar feel).
    pub fn siblings(self) -> &'static [Func] {
        match self {
            Func::Sin => &[Func::Cos],
            Func::Cos => &[Func::Sin],
            Func::Tan => &[Func::Tanh, Func::Sin],
            Func::Tanh => &[Func::Tan, Func::Sin],
            Func::Abs => &[Func::Fract, Func::Sqrt],
            Func::Fract => &[Func::Abs],
            Func::Sqrt => &[Func::Abs],
            Func::Floor => &[Func::Fract],
            Func::Min => &[Func::Max],
            Func::Max => &[Func::Min],
            Func::Exp => &[Func::Tanh],
            Func::Atan2 => &[Func::Mag],
            Func::Mag => &[Func::Atan2],
            _ => &[],
        }
    }
}

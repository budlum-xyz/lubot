#![forbid(unsafe_code)]
//! # lubot-tools - the things a model must not guess
//!
//! A language model answers `74830 * 1291` with a number of about the right
//! length and the wrong digits, because "likely continuation" and "correct
//! product" are different objectives. So arithmetic never reaches the model:
//! [`route`] recognises it, [`eval`] computes it, and the model is left to do
//! the part it is actually good at.
//!
//! The calculator works in exact rationals over `i128`. Floating point would
//! reintroduce, in the tool, the class of error the tool exists to remove:
//! `0.1 + 0.2` is `0.3` here, `1/3` prints as `1/3`, and an overflow is an
//! error rather than a silent wrap.

use std::fmt;

/// An exact rational, always kept in lowest terms with a positive denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    num: i128,
    den: i128,
}

impl Rational {
    /// Build a rational.
    ///
    /// # Errors
    /// Returns a description when the denominator is zero.
    pub fn new(num: i128, den: i128) -> Result<Self, String> {
        if den == 0 {
            return Err("division by zero".to_string());
        }
        let sign = if den < 0 { -1 } else { 1 };
        let (num, den) = (num * sign, den * sign);
        let g = gcd(num.unsigned_abs(), den.unsigned_abs());
        let g = if g == 0 { 1 } else { g as i128 };
        Ok(Self {
            num: num / g,
            den: den / g,
        })
    }

    #[must_use]
    pub fn whole(n: i128) -> Self {
        Self { num: n, den: 1 }
    }

    #[must_use]
    pub fn numerator(self) -> i128 {
        self.num
    }

    #[must_use]
    pub fn denominator(self) -> i128 {
        self.den
    }

    /// # Errors
    /// Returns a description on overflow.
    pub fn checked_add(self, other: Self) -> Result<Self, String> {
        let a = mul(self.num, other.den)?;
        let b = mul(other.num, self.den)?;
        Rational::new(add(a, b)?, mul(self.den, other.den)?)
    }

    /// # Errors
    /// Returns a description on overflow.
    pub fn checked_sub(self, other: Self) -> Result<Self, String> {
        let a = mul(self.num, other.den)?;
        let b = mul(other.num, self.den)?;
        Rational::new(sub(a, b)?, mul(self.den, other.den)?)
    }

    /// # Errors
    /// Returns a description on overflow.
    pub fn checked_mul(self, other: Self) -> Result<Self, String> {
        Rational::new(mul(self.num, other.num)?, mul(self.den, other.den)?)
    }

    /// # Errors
    /// Returns a description on overflow or division by zero.
    pub fn checked_div(self, other: Self) -> Result<Self, String> {
        if other.num == 0 {
            return Err("division by zero".to_string());
        }
        Rational::new(mul(self.num, other.den)?, mul(self.den, other.num)?)
    }

    /// Integer powers only: a fractional exponent is generally irrational, and
    /// an exact calculator that rounds is a calculator that lies quietly.
    ///
    /// # Errors
    /// Returns a description on overflow, on a non-integer exponent, or on a
    /// negative power of zero.
    pub fn checked_pow(self, exp: Self) -> Result<Self, String> {
        if exp.den != 1 {
            return Err("only whole exponents are exact; a root is not computed here".to_string());
        }
        let mut e = exp.num;
        let invert = e < 0;
        if invert && self.num == 0 {
            return Err("division by zero".to_string());
        }
        e = e.abs();
        if e > 64 {
            return Err("exponent too large to stay exact".to_string());
        }
        let mut acc = Rational::whole(1);
        for _ in 0..e {
            acc = acc.checked_mul(self)?;
        }
        if invert {
            return Rational::whole(1).checked_div(acc);
        }
        Ok(acc)
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            return write!(f, "{}", self.num);
        }
        // An exact decimal exists only when the denominator is 2/5-smooth.
        let mut d = self.den;
        let mut twos = 0u32;
        let mut fives = 0u32;
        while d % 2 == 0 {
            d /= 2;
            twos += 1;
        }
        while d % 5 == 0 {
            d /= 5;
            fives += 1;
        }
        if d != 1 {
            return write!(f, "{}/{}", self.num, self.den);
        }
        let places = twos.max(fives);
        let scale = 10i128.pow(places);
        let scaled = self.num * (scale / self.den);
        let sign = if scaled < 0 { "-" } else { "" };
        let abs = scaled.abs();
        let unit = abs / scale;
        let frac = abs % scale;
        let frac = format!("{frac:0width$}", width = places as usize);
        let frac = frac.trim_end_matches('0');
        if frac.is_empty() {
            return write!(f, "{sign}{unit}");
        }
        write!(f, "{sign}{unit}.{frac}")
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn mul(a: i128, b: i128) -> Result<i128, String> {
    a.checked_mul(b).ok_or_else(|| "overflow".to_string())
}

fn add(a: i128, b: i128) -> Result<i128, String> {
    a.checked_add(b).ok_or_else(|| "overflow".to_string())
}

fn sub(a: i128, b: i128) -> Result<i128, String> {
    a.checked_sub(b).ok_or_else(|| "overflow".to_string())
}

/// Where a question should go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// The calculator answered it; the model reports this string.
    Tool { name: &'static str, result: String },
    /// The calculator recognised arithmetic and refused it, with a reason.
    ToolFailed { name: &'static str, reason: String },
    /// Not arithmetic: the reading path handles it.
    Model,
}

/// Decide where a question goes, and answer it here when it is arithmetic.
#[must_use]
pub fn route(question: &str) -> Route {
    let Some(expr) = arithmetic_part(question) else {
        return Route::Model;
    };
    match eval(&expr) {
        Ok(value) => Route::Tool {
            name: "calculator",
            result: value.to_string(),
        },
        Err(reason) => Route::ToolFailed {
            name: "calculator",
            reason,
        },
    }
}

/// Pull the arithmetic out of a sentence, or `None` when there is none.
///
/// The rule is deliberately narrow: a run of digits, operators, spaces and
/// parentheses containing at least one operator and one digit. A wider rule
/// would send prose to a calculator, and the calculator would answer.
fn arithmetic_part(question: &str) -> Option<String> {
    let mut best = String::new();
    let mut current = String::new();
    for ch in question.chars() {
        if ch.is_ascii_digit() || "+-*/^(). ".contains(ch) {
            current.push(ch);
        } else {
            if score(&current) > score(&best) {
                best = current.clone();
            }
            current.clear();
        }
    }
    if score(&current) > score(&best) {
        best = current;
    }
    let trimmed = best.trim().to_string();
    if score(&trimmed) == 0 {
        return None;
    }
    Some(trimmed)
}

fn score(s: &str) -> usize {
    let has_digit = s.chars().any(|c| c.is_ascii_digit());
    let has_op = s.chars().any(|c| "+-*/^".contains(c));
    if has_digit && has_op {
        s.trim().len()
    } else {
        0
    }
}

/// Evaluate an arithmetic expression exactly.
///
/// # Errors
/// Returns a description for a malformed expression, a division by zero or an
/// overflow.
pub fn eval(expr: &str) -> Result<Rational, String> {
    let tokens = lex(expr)?;
    let mut parser = Parser { tokens, pos: 0 };
    let value = parser.expression(0)?;
    if parser.pos != parser.tokens.len() {
        return Err(format!(
            "the expression does not parse from end to end: stopped at token {}",
            parser.pos
        ));
    }
    Ok(value)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Number(Rational),
    Op(char),
    Open,
    Close,
}

fn lex(expr: &str) -> Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let chars: Vec<char> = expr.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            ' ' => i += 1,
            '(' => {
                out.push(Token::Open);
                i += 1;
            }
            ')' => {
                out.push(Token::Close);
                i += 1;
            }
            '+' | '-' | '*' | '/' | '^' => {
                out.push(Token::Op(c));
                i += 1;
            }
            d if d.is_ascii_digit() || d == '.' => {
                let start = i;
                let mut dots = 0;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    if chars[i] == '.' {
                        dots += 1;
                    }
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                if dots > 1 {
                    return Err(format!("`{text}` is not a number"));
                }
                out.push(Token::Number(parse_decimal(&text)?));
            }
            other => return Err(format!("`{other}` is not part of an arithmetic expression")),
        }
    }
    if out.is_empty() {
        return Err("there is nothing to compute".to_string());
    }
    Ok(out)
}

fn parse_decimal(text: &str) -> Result<Rational, String> {
    let (whole, frac) = match text.split_once('.') {
        Some((w, f)) => (w, f),
        None => (text, ""),
    };
    let whole = if whole.is_empty() { "0" } else { whole };
    let w: i128 = whole
        .parse()
        .map_err(|_| format!("`{text}` does not fit an exact integer"))?;
    if frac.is_empty() {
        return Ok(Rational::whole(w));
    }
    let f: i128 = frac
        .parse()
        .map_err(|_| format!("`{text}` does not fit an exact integer"))?;
    let scale = 10i128
        .checked_pow(u32::try_from(frac.len()).map_err(|_| "number too long".to_string())?)
        .ok_or_else(|| "number too long".to_string())?;
    Rational::new(w * scale + f, scale)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn expression(&mut self, min_bp: u8) -> Result<Rational, String> {
        let mut lhs = self.atom()?;
        while let Some(Token::Op(op)) = self.tokens.get(self.pos).cloned() {
            let (l_bp, r_bp) = match op {
                '+' | '-' => (1, 2),
                '*' | '/' => (3, 4),
                // Right-associative: 2^3^2 is 2^9, not 8^2.
                '^' => (6, 5),
                _ => return Err(format!("`{op}` is not an operator")),
            };
            if l_bp < min_bp {
                break;
            }
            self.pos += 1;
            let rhs = self.expression(r_bp)?;
            lhs = match op {
                '+' => lhs.checked_add(rhs)?,
                '-' => lhs.checked_sub(rhs)?,
                '*' => lhs.checked_mul(rhs)?,
                '/' => lhs.checked_div(rhs)?,
                '^' => lhs.checked_pow(rhs)?,
                _ => return Err(format!("`{op}` is not an operator")),
            };
        }
        Ok(lhs)
    }

    fn atom(&mut self) -> Result<Rational, String> {
        match self.tokens.get(self.pos).cloned() {
            Some(Token::Number(n)) => {
                self.pos += 1;
                Ok(n)
            }
            Some(Token::Op('-')) => {
                self.pos += 1;
                let inner = self.expression(5)?;
                Rational::whole(0).checked_sub(inner)
            }
            Some(Token::Op('+')) => {
                self.pos += 1;
                self.expression(5)
            }
            Some(Token::Open) => {
                self.pos += 1;
                let inner = self.expression(0)?;
                match self.tokens.get(self.pos) {
                    Some(Token::Close) => {
                        self.pos += 1;
                        Ok(inner)
                    }
                    _ => Err("a parenthesis is opened and never closed".to_string()),
                }
            }
            _ => Err("an operand is missing".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(expr: &str) -> String {
        eval(expr).map(|v| v.to_string()).unwrap_or_else(|e| e)
    }

    #[test]
    fn the_product_a_model_gets_wrong() {
        assert_eq!(value("74830 * 1291"), "96605530");
    }

    #[test]
    fn decimals_are_exact_not_floating() {
        assert_eq!(value("0.1 + 0.2"), "0.3");
        assert_eq!(value("1.005 * 100"), "100.5");
    }

    #[test]
    fn a_repeating_fraction_is_printed_as_a_fraction() {
        assert_eq!(value("1 / 3"), "1/3");
        assert_eq!(value("2 / 4"), "0.5");
    }

    #[test]
    fn precedence_and_parentheses_hold() {
        assert_eq!(value("2 + 3 * 4"), "14");
        assert_eq!(value("(2 + 3) * 4"), "20");
        assert_eq!(value("-3 + 5"), "2");
    }

    #[test]
    fn exponentiation_is_right_associative() {
        assert_eq!(value("2 ^ 3 ^ 2"), "512");
        assert_eq!(value("2 ^ -2"), "0.25");
    }

    #[test]
    fn division_by_zero_is_an_error_not_an_infinity() {
        assert_eq!(value("1 / 0"), "division by zero");
        assert_eq!(value("0 ^ -1"), "division by zero");
    }

    #[test]
    fn overflow_is_refused_rather_than_wrapped() {
        let err = eval("170141183460469231731687303715884105727 * 2").unwrap_err();
        assert!(err.contains("overflow") || err.contains("exact integer"));
    }

    #[test]
    fn a_fractional_exponent_is_refused_instead_of_rounded() {
        let err = eval("2 ^ 0.5").unwrap_err();
        assert!(err.contains("whole exponents"));
    }

    #[test]
    fn malformed_input_is_refused() {
        assert!(eval("2 +").is_err());
        assert!(eval("(2 + 3").is_err());
        assert!(eval("").is_err());
    }

    #[test]
    fn arithmetic_is_routed_to_the_tool() {
        assert_eq!(
            route("74830 * 1291 kac eder?"),
            Route::Tool {
                name: "calculator",
                result: "96605530".to_string()
            }
        );
    }

    #[test]
    fn a_question_without_arithmetic_goes_to_the_reading_path() {
        assert_eq!(route("what does a view grant do?"), Route::Model);
        assert_eq!(route("version 3 of the format"), Route::Model);
    }

    #[test]
    fn a_recognised_but_impossible_computation_reports_the_reason() {
        match route("what is 1 / 0") {
            Route::ToolFailed { reason, .. } => assert!(reason.contains("division by zero")),
            other => panic!("expected a tool failure, got {other:?}"),
        }
    }

    #[test]
    fn rationals_are_kept_in_lowest_terms() {
        let r = Rational::new(6, 8).unwrap();
        assert_eq!(r.numerator(), 3);
        assert_eq!(r.denominator(), 4);
        let neg = Rational::new(1, -2).unwrap();
        assert_eq!(neg.numerator(), -1);
        assert_eq!(neg.denominator(), 2);
    }
}

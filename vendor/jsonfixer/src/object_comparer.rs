use serde_json::Value;

pub struct ObjectComparer;

impl ObjectComparer {
    pub fn is_same_object(obj1: &Value, obj2: &Value) -> bool {
        std::mem::discriminant(obj1) == std::mem::discriminant(obj2)
            && match (obj1, obj2) {
                (Value::Object(map1), Value::Object(map2)) => {
                    map1.len() == map2.len()
                        && map1.iter().all(|(key, value1)| {
                            map2.get(key).map_or(false, |value2| {
                                Self::is_same_object(value1, value2)
                            })
                        })
                }
                (Value::Array(arr1), Value::Array(arr2)) => {
                    arr1.len() == arr2.len()
                        && arr1.iter().zip(arr2.iter()).all(|(v1, v2)| {
                            Self::is_same_object(v1, v2)
                        })
                }
                (Value::String(s1), Value::String(s2)) => s1 == s2,
                (Value::Number(n1), Value::Number(n2)) => n1 == n2,
                (Value::Bool(b1), Value::Bool(b2)) => b1 == b2,
                (Value::Null, Value::Null) => true,
                _ => false,
            }
    }

    pub fn is_strictly_empty(value: &Value) -> bool {
        match value {
            Value::String(s) => s.is_empty(),
            Value::Array(arr) => arr.is_empty(),
            Value::Object(map) => map.is_empty(),
            _ => false,
        }
    }
}
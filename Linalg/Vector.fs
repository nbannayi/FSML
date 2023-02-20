namespace ML.Linalg

/// Represents a vector with any number of dimensions.
type Vector = double list

/// Vector module for manipulation of Vector types.
module Vector =

    /// Add together two vectors.
    let add (v: Vector) (w: Vector) =
        match v, w with
        | v, w when v.Length <> w.Length ->
            failwith "Vectors must be the same length."
        | _ ->
            List.zip v w |> List.map (fun (v',w') -> v'+w')

    /// Subtract second passed vector from first.
    let subtract (v: Vector) (w: Vector) =
        match v, w with
        | v, w when v.Length <> w.Length ->
            failwith "Vectors must be the same length."
        | _ ->
            List.zip v w |> List.map (fun (v',w') -> v'-w')

    /// Sums all corresponding elements of a list of vectors.
    let vectorsum (vectors: Vector list) =
        match vectors with
        | [] ->
            failwith "No vectors provided."
        | vectors when let n = vectors.[0].Length in vectors |> List.forall (fun v -> v.Length = n) = false ->
            failwith "Vectors must all be the same length."
        | _ ->
            let l, n = vectors.Length-1, vectors.[0].Length-1
            [for n' in [0..n] do seq {for l' in [0..l] do yield vectors.[l'].[n']}]
            |> List.map (fun s -> s |> Seq.sum)

    /// Multiplies every element of a vector by the scalar c.
    let scalarmul (c: double) (v: Vector) =
        v |> List.map (fun e -> c*e)

    /// Returns the element-wise average of a list of vectors.
    let vectormean (vectors: Vector list) =
        let n = vectors.Length
        scalarmul (1./(double n)) (vectorsum vectors)

    /// Computes vector dot product (i.e. v_t = v_1 * w_1 + ... + v_n * w_n.) 
    let dot (v: Vector) (w: Vector) =
        match v, w with
        | v, w when v.Length <> w.Length ->
            failwith "Vectors must be the same length."
        | _ ->
            List.zip v w
            |> List.map (fun (v',w') -> v'*w')
            |> List.sum
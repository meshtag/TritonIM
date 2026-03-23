; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i32 @__pim_get_bank_id()

declare i32 @__pim_get_program_id()

define void @axpy_kernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2, i32 %3, ptr addrspace(1) %4, ptr addrspace(1) %5) {
  %7 = call i32 @__pim_get_program_id()
  %8 = mul i32 %7, 64
  %9 = call i32 @__pim_get_bank_id()
  %10 = and i32 %9, 15
  %11 = shl i32 %10, 0
  %12 = or i32 0, %11
  %13 = and i32 %12, 15
  %14 = shl i32 %13, 2
  %15 = or disjoint i32 %14, 0
  %16 = xor i32 0, %15
  %17 = xor i32 %16, 0
  %18 = xor i32 %16, 1
  %19 = xor i32 %16, 2
  %20 = xor i32 %16, 3
  %21 = add i32 %17, 0
  %22 = add i32 %18, 0
  %23 = add i32 %19, 0
  %24 = add i32 %20, 0
  %25 = add i32 %8, %21
  %26 = add i32 %8, %22
  %27 = add i32 %8, %23
  %28 = add i32 %8, %24
  %29 = icmp slt i32 %25, %3
  %30 = icmp slt i32 %26, %3
  %31 = icmp slt i32 %27, %3
  %32 = icmp slt i32 %28, %3
  %33 = getelementptr i32, ptr addrspace(1) %0, i32 %25
  %34 = getelementptr i32, ptr addrspace(1) %0, i32 %26
  %35 = getelementptr i32, ptr addrspace(1) %0, i32 %27
  %36 = getelementptr i32, ptr addrspace(1) %0, i32 %28
  br i1 %29, label %37, label %39

37:                                               ; preds = %6
  %38 = load i32, ptr addrspace(1) %33, align 4
  br label %39

39:                                               ; preds = %37, %6
  %40 = phi i32 [ %38, %37 ], [ 0, %6 ]
  br i1 %30, label %41, label %43

41:                                               ; preds = %39
  %42 = load i32, ptr addrspace(1) %34, align 4
  br label %43

43:                                               ; preds = %41, %39
  %44 = phi i32 [ %42, %41 ], [ 0, %39 ]
  br i1 %31, label %45, label %47

45:                                               ; preds = %43
  %46 = load i32, ptr addrspace(1) %35, align 4
  br label %47

47:                                               ; preds = %45, %43
  %48 = phi i32 [ %46, %45 ], [ 0, %43 ]
  br i1 %32, label %49, label %51

49:                                               ; preds = %47
  %50 = load i32, ptr addrspace(1) %36, align 4
  br label %51

51:                                               ; preds = %49, %47
  %52 = phi i32 [ %50, %49 ], [ 0, %47 ]
  %53 = getelementptr i32, ptr addrspace(1) %1, i32 %25
  %54 = getelementptr i32, ptr addrspace(1) %1, i32 %26
  %55 = getelementptr i32, ptr addrspace(1) %1, i32 %27
  %56 = getelementptr i32, ptr addrspace(1) %1, i32 %28
  br i1 %29, label %57, label %59

57:                                               ; preds = %51
  %58 = load i32, ptr addrspace(1) %53, align 4
  br label %59

59:                                               ; preds = %57, %51
  %60 = phi i32 [ %58, %57 ], [ 0, %51 ]
  br i1 %30, label %61, label %63

61:                                               ; preds = %59
  %62 = load i32, ptr addrspace(1) %54, align 4
  br label %63

63:                                               ; preds = %61, %59
  %64 = phi i32 [ %62, %61 ], [ 0, %59 ]
  br i1 %31, label %65, label %67

65:                                               ; preds = %63
  %66 = load i32, ptr addrspace(1) %55, align 4
  br label %67

67:                                               ; preds = %65, %63
  %68 = phi i32 [ %66, %65 ], [ 0, %63 ]
  br i1 %32, label %69, label %71

69:                                               ; preds = %67
  %70 = load i32, ptr addrspace(1) %56, align 4
  br label %71

71:                                               ; preds = %69, %67
  %72 = phi i32 [ %70, %69 ], [ 0, %67 ]
  %73 = mul i32 %2, %40
  %74 = mul i32 %2, %44
  %75 = mul i32 %2, %48
  %76 = mul i32 %2, %52
  %77 = add i32 %73, %60
  %78 = add i32 %74, %64
  %79 = add i32 %75, %68
  %80 = add i32 %76, %72
  br i1 %29, label %81, label %82

81:                                               ; preds = %71
  store i32 %77, ptr addrspace(1) %53, align 4
  br label %82

82:                                               ; preds = %81, %71
  br i1 %30, label %83, label %84

83:                                               ; preds = %82
  store i32 %78, ptr addrspace(1) %54, align 4
  br label %84

84:                                               ; preds = %83, %82
  br i1 %31, label %85, label %86

85:                                               ; preds = %84
  store i32 %79, ptr addrspace(1) %55, align 4
  br label %86

86:                                               ; preds = %85, %84
  br i1 %32, label %87, label %88

87:                                               ; preds = %86
  store i32 %80, ptr addrspace(1) %56, align 4
  br label %88

88:                                               ; preds = %87, %86
  ret void
}


!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

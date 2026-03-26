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
  br i1 %29, label %34, label %44

34:                                               ; preds = %6
  %35 = load <4 x i32>, ptr addrspace(1) %33, align 16
  %36 = extractelement <4 x i32> %35, i32 0
  %37 = select i1 %29, i32 %36, i32 0
  %38 = extractelement <4 x i32> %35, i32 1
  %39 = select i1 %30, i32 %38, i32 0
  %40 = extractelement <4 x i32> %35, i32 2
  %41 = select i1 %31, i32 %40, i32 0
  %42 = extractelement <4 x i32> %35, i32 3
  %43 = select i1 %32, i32 %42, i32 0
  br label %44

44:                                               ; preds = %34, %6
  %45 = phi i32 [ %37, %34 ], [ 0, %6 ]
  %46 = phi i32 [ %39, %34 ], [ 0, %6 ]
  %47 = phi i32 [ %41, %34 ], [ 0, %6 ]
  %48 = phi i32 [ %43, %34 ], [ 0, %6 ]
  %49 = getelementptr i32, ptr addrspace(1) %1, i32 %25
  %50 = getelementptr i32, ptr addrspace(1) %1, i32 %26
  %51 = getelementptr i32, ptr addrspace(1) %1, i32 %27
  %52 = getelementptr i32, ptr addrspace(1) %1, i32 %28
  br i1 %29, label %53, label %63

53:                                               ; preds = %44
  %54 = load <4 x i32>, ptr addrspace(1) %49, align 16
  %55 = extractelement <4 x i32> %54, i32 0
  %56 = select i1 %29, i32 %55, i32 0
  %57 = extractelement <4 x i32> %54, i32 1
  %58 = select i1 %30, i32 %57, i32 0
  %59 = extractelement <4 x i32> %54, i32 2
  %60 = select i1 %31, i32 %59, i32 0
  %61 = extractelement <4 x i32> %54, i32 3
  %62 = select i1 %32, i32 %61, i32 0
  br label %63

63:                                               ; preds = %53, %44
  %64 = phi i32 [ %56, %53 ], [ 0, %44 ]
  %65 = phi i32 [ %58, %53 ], [ 0, %44 ]
  %66 = phi i32 [ %60, %53 ], [ 0, %44 ]
  %67 = phi i32 [ %62, %53 ], [ 0, %44 ]
  %68 = mul i32 %2, %45
  %69 = mul i32 %2, %46
  %70 = mul i32 %2, %47
  %71 = mul i32 %2, %48
  %72 = add i32 %68, %64
  %73 = add i32 %69, %65
  %74 = add i32 %70, %66
  %75 = add i32 %71, %67
  %76 = and i1 %29, %30
  %77 = and i1 %76, %31
  %78 = and i1 %77, %32
  %79 = insertelement <4 x i32> undef, i32 %72, i32 0
  %80 = insertelement <4 x i32> %79, i32 %73, i32 1
  %81 = insertelement <4 x i32> %80, i32 %74, i32 2
  %82 = insertelement <4 x i32> %81, i32 %75, i32 3
  br i1 %78, label %83, label %84

83:                                               ; preds = %63
  store <4 x i32> %82, ptr addrspace(1) %49, align 16
  br label %92

84:                                               ; preds = %63
  br i1 %29, label %85, label %86

85:                                               ; preds = %84
  store i32 %72, ptr addrspace(1) %49, align 4
  br label %86

86:                                               ; preds = %85, %84
  br i1 %30, label %87, label %88

87:                                               ; preds = %86
  store i32 %73, ptr addrspace(1) %50, align 4
  br label %88

88:                                               ; preds = %87, %86
  br i1 %31, label %89, label %90

89:                                               ; preds = %88
  store i32 %74, ptr addrspace(1) %51, align 4
  br label %90

90:                                               ; preds = %89, %88
  br i1 %32, label %91, label %92

91:                                               ; preds = %90
  store i32 %75, ptr addrspace(1) %52, align 4
  br label %92

92:                                               ; preds = %83, %91, %90
  ret void
}


!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}

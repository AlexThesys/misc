#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <cstdio>
#include <vector>

bool get_windows_version()
{
    OSVERSIONINFOEXW osInfo = { 0 };
    osInfo.dwOSVersionInfoSize = sizeof(osInfo);

    typedef LONG(WINAPI* RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
    RtlGetVersionPtr RtlGetVersion = (RtlGetVersionPtr)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion");

    if (RtlGetVersion == nullptr) {
        perror("Failed to locate RtlGetVersion function.");
        return false;
    }
    if (RtlGetVersion(reinterpret_cast<PRTL_OSVERSIONINFOW>(&osInfo)) != 0) {
        perror("Failed to retrieve version information.");
        return false;
    }

    const bool is_server = (osInfo.wProductType == VER_NT_SERVER || osInfo.wProductType == VER_NT_DOMAIN_CONTROLLER);
    if (osInfo.dwMajorVersion == 10) {
        if (is_server) {
            if (osInfo.dwBuildNumber >= 20348) {
                puts("Windows Server 2022 detected.");
                return true;
            } else if (osInfo.dwBuildNumber >= 17763) {
                puts("Windows Server 2019 detected.");
            } else {
                puts("An older version of Windows Server detected (Windows Server 2016 or earlier).");
            }
        } else {
            if (osInfo.dwBuildNumber >= 22000) {
                puts("Windows 11 detected.");
                return true;
            }
            else {
                puts("Windows 10 detected.");
            }
        }
    }
    else {
        puts("An unsupported Windows version detected.");
    }

    printf("Version: %d.%d\n", osInfo.dwMajorVersion, osInfo.dwMinorVersion);
    printf("Build Number: %d\n", osInfo.dwBuildNumber);
    return false;
}

void get_processor_info(WORD target_group)
{
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size);

    if (buffer_size == 0) {
        perror("Failed to get buffer size for processor information.");
        return;
    }

    // Allocate the buffer
    std::vector<BYTE> buffer(buffer_size);

    // Second call to actually retrieve the information
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
       perror("Failed to get processor information.");
        return;
    }

    int p_cores = 0;
    int e_cores = 0;
    int logical_processor_count = 0;

    // Traverse the buffer and count cores and logical processors only for the specified group
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());

    const bool query_all_processor_groups = (target_group == ALL_PROCESSOR_GROUPS);

    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            // Check if the processor's group matches the target group
            if (query_all_processor_groups || ((info->Processor.GroupCount > 0) && (info->Processor.GroupMask[0].Group == target_group))) {
                DWORD_PTR mask = info->Processor.GroupMask[0].Mask;
                int lp_count = 0;
                for (; mask; mask &= (mask - 1)) {
                    lp_count++;
                }
                // Count the number of bits set in the GroupMask for logical processors
                if (lp_count) {
                    logical_processor_count += lp_count;
                    if (info->Processor.EfficiencyClass == 0) {
                        e_cores++;
                    } else {
                        p_cores++;
                    }
                }
            }
        }

        // Move to the next SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX structure
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<BYTE*>(info) + info->Size);
    }

    if (!p_cores) {
        p_cores = e_cores;
        e_cores = 0;
    }

    printf("Physical p-cores in group #0x%x : %d\n", target_group, p_cores);
    printf("Physical e-cores in group  #0x%x : %d\n", target_group, e_cores);
    printf("Logical processors in group  #0x%x : %d\n", target_group, logical_processor_count);
}

DWORD get_current_processor_group(DWORD num_cur_group_cpus)
{
    DWORD result = 0;
    const DWORD active_group_count = GetActiveProcessorGroupCount();
    printf("Number of active processor groups: %d\n", active_group_count);
    if (active_group_count < 2) {
        return result;
    }
    for (DWORD i = 0; i < active_group_count; i++) {
        const DWORD num_group_cpus = GetActiveProcessorCount(i);
        printf("Logical processors in group  #%d : %d\n", i, num_group_cpus);
        if (num_cur_group_cpus == num_group_cpus) {
            result = i;
            break;
        }
    }

    return result;
}

int main()
{
    const bool windows11 = get_windows_version();
    DWORD current_group = 0;
    if (!windows11) {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        current_group = get_current_processor_group(sys_info.dwNumberOfProcessors);
    } else {
        current_group = ALL_PROCESSOR_GROUPS;
    }
    get_processor_info((WORD)current_group);
    return 0;
}